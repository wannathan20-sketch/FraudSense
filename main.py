#!/usr/bin/env python3
"""
信用卡欺诈检测：完整训练和评估流程。

默认使用全量数据训练所有模型。开发或演示时可以使用 --quick 快速跑通流程。
"""
import argparse
import os

# 在导入数值计算库前限制线程数，减少 macOS 上的 BLAS 线程问题。
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import warnings
import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.model_selection import train_test_split as sklearn_train_test_split

import config
from data.dataset import load_raw, split_and_preprocess, to_tensor, get_dataloader
from models.autoencoder import Autoencoder
from train.evaluate import (
    compute_metrics,
    compute_metrics_from_validation_threshold,
    find_optimal_threshold,
    format_metrics_table,
    save_metrics,
)
from train.supervised import make_lgb_classifier, make_xgb_classifier
from train.train_ae import train_autoencoder
from utils.metrics import precision_at_k, recall_at_k
from utils.visualization import (
    plot_confusion_matrix,
    plot_latent_space,
    plot_metrics_comparison,
    plot_pr_curve,
    plot_recon_error_distribution,
    plot_roc_curve,
    plot_training_curves,
)


warnings.filterwarnings("ignore")


def build_arg_parser():
    parser = argparse.ArgumentParser(
        description="训练并评估信用卡欺诈检测模型。"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--quick",
        action="store_true",
        help="使用较小样本快速跑通流程。",
    )
    mode.add_argument(
        "--full",
        action="store_true",
        help="使用全量数据运行完整流程，这是默认模式。",
    )
    parser.add_argument(
        "--quick-sample-size",
        type=int,
        default=30000,
        help="quick 模式下使用的分层抽样行数。",
    )
    parser.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help="图表和指标文件的输出目录。",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="跳过图表生成，用于更快的快速验收。",
    )
    return parser


def stratified_sample(df, sample_size):
    if sample_size >= len(df):
        return df.copy()
    sample, _ = sklearn_train_test_split(
        df,
        train_size=sample_size,
        random_state=config.RANDOM_STATE,
        stratify=df["Class"],
    )
    return sample.reset_index(drop=True)


def make_smote(y_train):
    fraud_count = int(y_train.sum())
    if fraud_count < 2:
        raise ValueError("SMOTE 至少需要 2 条训练集欺诈样本。")
    k_neighbors = min(5, fraud_count - 1)
    return SMOTE(random_state=config.RANDOM_STATE, k_neighbors=k_neighbors)


def print_metric_line(label, metrics):
    print(
        f"  {label:<4} |  AUPRC: {metrics['auprc']:.4f}  |  "
        f"AUC-ROC: {metrics['auc_roc']:.4f}  |  "
        f"F1: {metrics['f1']:.4f}  |  "
        f"Precision: {metrics['precision']:.4f}  |  "
        f"Recall: {metrics['recall']:.4f}"
    )


def run_pipeline(args=None):
    args = args or build_arg_parser().parse_args()

    config.OUTPUT_DIR = os.path.abspath(args.output_dir)
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mode_name = "quick" if args.quick else "full"
    ae_epochs = 5 if args.quick else config.AE_EPOCHS
    ae_patience = 3 if args.quick else config.AE_PATIENCE
    tree_estimators = 30 if args.quick else None
    tsne_samples = 1000 if args.quick else 5000

    print(f"使用设备: {device}")
    print(f"运行模式: {mode_name}")

    print("\n" + "=" * 60)
    print("1. 读取和预处理数据")
    print("=" * 60)

    df = load_raw()
    if args.quick:
        df = stratified_sample(df, args.quick_sample_size)
        print(f"  quick 抽样已启用: {len(df):,} 行")

    print(f"  样本总数: {len(df):,}")
    print(f"  欺诈样本: {df['Class'].sum():,} ({df['Class'].mean()*100:.3f}%)")
    print(f"  特征数: {df.shape[1] - 1}")

    X_train, X_val, X_test, y_train, y_val, y_test, _ = split_and_preprocess(df)

    print(
        f"  训练集: {len(X_train):,}  |  验证集: {len(X_val):,}  |  "
        f"测试集: {len(X_test):,}"
    )
    print(
        f"  训练集欺诈: {y_train.sum():.0f}  |  "
        f"验证集欺诈: {y_val.sum():.0f}  |  测试集欺诈: {y_test.sum():.0f}"
    )

    X_train_normal = X_train[y_train == 0]
    print(f"  正常交易训练样本: {len(X_train_normal):,}")

    X_test_tensor = to_tensor(X_test).to(device)
    X_val_tensor = to_tensor(X_val).to(device)

    print("\n" + "=" * 60)
    print("2. 训练 Autoencoder（只用正常交易）")
    print("=" * 60)

    n_ae_train = int(len(X_train_normal) * 0.8)
    X_ae_train = X_train_normal[:n_ae_train]
    X_ae_val = X_train_normal[n_ae_train:]

    ae_train_loader = get_dataloader(X_ae_train, shuffle=True)
    ae_val_loader = get_dataloader(X_ae_val, shuffle=False)

    ae_model = Autoencoder(
        input_dim=X_train.shape[1],
        hidden_dims=config.AE_HIDDEN_DIMS,
        latent_dim=config.AE_LATENT_DIM,
        dropout=config.AE_DROPOUT,
    ).to(device)

    ae_model, ae_history = train_autoencoder(
        ae_model,
        ae_train_loader,
        ae_val_loader,
        epochs=ae_epochs,
        patience=ae_patience,
    )
    if not args.skip_plots:
        plot_training_curves(ae_history, "AE")

    ae_model.eval()
    with torch.no_grad():
        ae_recon_errors = ae_model.reconstruction_error(X_test_tensor).cpu().numpy()
        ae_val_errors = ae_model.reconstruction_error(X_val_tensor).cpu().numpy()

    ae_threshold, _ = find_optimal_threshold(ae_val_errors, y_val)
    ae_metrics = compute_metrics(
        ae_recon_errors,
        y_test,
        threshold=ae_threshold,
        method_name="Autoencoder",
    )
    print_metric_line("AE", ae_metrics)

    if not args.skip_plots:
        with torch.no_grad():
            ae_z, _ = ae_model(X_test_tensor)
        plot_latent_space(
            ae_z.cpu().numpy(),
            y_test,
            "Autoencoder",
            max_samples=tsne_samples,
        )
        plot_recon_error_distribution(
            ae_recon_errors[y_test == 0],
            ae_recon_errors[y_test == 1],
            "Autoencoder",
        )

    del ae_model, ae_train_loader, ae_val_loader, X_ae_train, X_ae_val
    if device.type == "cuda":
        torch.cuda.empty_cache()

    print("\n" + "=" * 60)
    print("3. PCA 重构误差基线")
    print("=" * 60)

    pca = PCA(n_components=min(X_train_normal.shape[1], 10))
    pca.fit(X_train_normal)

    X_test_pca_recon = pca.inverse_transform(pca.transform(X_test))
    pca_recon_errors = np.mean((X_test - X_test_pca_recon) ** 2, axis=1)

    X_val_pca_recon = pca.inverse_transform(pca.transform(X_val))
    pca_val_errors = np.mean((X_val - X_val_pca_recon) ** 2, axis=1)

    pca_threshold, _ = find_optimal_threshold(pca_val_errors, y_val)
    pca_metrics = compute_metrics(
        pca_recon_errors,
        y_test,
        threshold=pca_threshold,
        method_name="PCA-Recon",
    )
    print_metric_line("PCA", pca_metrics)

    print("\n" + "=" * 60)
    print("4. Isolation Forest")
    print("=" * 60)

    n_if_train = min(50000, len(X_train_normal))
    rng = np.random.RandomState(config.RANDOM_STATE)
    if_indices = rng.choice(len(X_train_normal), n_if_train, replace=False)
    X_if_train = X_train_normal[if_indices]

    contamination = y_train.sum() / len(y_train)
    iso_forest = IsolationForest(
        n_estimators=100 if not args.quick else 50,
        contamination=contamination,
        random_state=config.RANDOM_STATE,
        n_jobs=1,
    )
    iso_forest.fit(X_if_train)

    if_scores = -iso_forest.decision_function(X_test)
    if_val_scores = -iso_forest.decision_function(X_val)

    if_threshold, _ = find_optimal_threshold(if_val_scores, y_val)
    if_metrics = compute_metrics(
        if_scores,
        y_test,
        threshold=if_threshold,
        method_name="IsolationForest",
    )
    print_metric_line("IF", if_metrics)

    print("\n" + "=" * 60)
    print("5. 监督式基线（XGBoost & LightGBM + SMOTE）")
    print("=" * 60)

    print("  应用 SMOTE...")
    smote = make_smote(y_train)
    X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
    print(
        f"  SMOTE 后: {len(X_train_smote):,} 条样本 "
        f"({y_train_smote.sum():.0f} 条欺诈)"
    )

    print("  训练 XGBoost...")
    xgb = make_xgb_classifier(n_estimators=tree_estimators)
    xgb.fit(X_train_smote, y_train_smote)
    xgb_val_probs = xgb.predict_proba(X_val)[:, 1]
    xgb_probs = xgb.predict_proba(X_test)[:, 1]
    xgb_metrics = compute_metrics_from_validation_threshold(
        xgb_val_probs,
        y_val,
        xgb_probs,
        y_test,
        method_name="XGBoost+SMOTE",
    )
    print_metric_line("XGB", xgb_metrics)

    print("  训练 LightGBM...")
    lgb = make_lgb_classifier(n_estimators=tree_estimators)
    lgb.fit(X_train_smote, y_train_smote)
    lgb_val_probs = lgb.predict_proba(X_val)[:, 1]
    lgb_probs = lgb.predict_proba(X_test)[:, 1]
    lgb_metrics = compute_metrics_from_validation_threshold(
        lgb_val_probs,
        y_val,
        lgb_probs,
        y_test,
        method_name="LightGBM+SMOTE",
    )
    print_metric_line("LGB", lgb_metrics)

    print("\n" + "=" * 60)
    print("6. 最终模型对比")
    print("=" * 60)

    all_metrics = [ae_metrics, pca_metrics, if_metrics, xgb_metrics, lgb_metrics]
    print(format_metrics_table(all_metrics))
    csv_path, json_path = save_metrics(all_metrics, config.OUTPUT_DIR)
    print(f"\n指标 CSV 已保存到: {csv_path}")
    print(f"指标 JSON 已保存到: {json_path}")

    print("\nPrecision@k / Recall@k（top-k 风险筛查）:")
    scores_dict = {
        "Autoencoder": ae_recon_errors,
        "PCA-Recon": pca_recon_errors,
        "IsolationForest": if_scores,
        "XGBoost+SMOTE": xgb_probs,
        "LightGBM+SMOTE": lgb_probs,
    }
    for name, scores in scores_dict.items():
        pk1 = precision_at_k(y_test, scores, k_ratio=0.01)
        rk1 = recall_at_k(y_test, scores, k_ratio=0.01)
        pk5 = precision_at_k(y_test, scores, k_ratio=0.05)
        rk5 = recall_at_k(y_test, scores, k_ratio=0.05)
        print(
            f"  {name:<20} | P@1%: {pk1:.4f}  R@1%: {rk1:.4f}  |  "
            f"P@5%: {pk5:.4f}  R@5%: {rk5:.4f}"
        )

    if not args.skip_plots:
        print("\n" + "=" * 60)
        print("7. 生成对比图表")
        print("=" * 60)

        plot_pr_curve(scores_dict, y_test)
        plot_roc_curve(scores_dict, y_test)

        ae_pred = (ae_recon_errors >= ae_threshold).astype(int)
        xgb_pred = (xgb_probs >= xgb_metrics["threshold"]).astype(int)
        plot_confusion_matrix(y_test, ae_pred, "Autoencoder")
        plot_confusion_matrix(y_test, xgb_pred, "XGBoost+SMOTE")
        plot_metrics_comparison(all_metrics)

    print(f"\n所有输出已保存到: {config.OUTPUT_DIR}")
    print("完成!")
    return all_metrics


def main():
    args = build_arg_parser().parse_args()
    run_pipeline(args)


if __name__ == "__main__":
    main()
