"""
欺诈检测项目的可视化工具。
所有图都会保存到 OUTPUT_DIR。
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, precision_recall_curve, roc_curve, auc
from sklearn.manifold import TSNE

import config

# 全局绘图风格
plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
})
sns.set_style("whitegrid")


def save_or_show(fig, name):
    """把图片保存到 OUTPUT_DIR。"""
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    path = os.path.join(config.OUTPUT_DIR, name)
    fig.savefig(path, bbox_inches="tight", dpi=150)
    plt.close(fig)


# ── 训练曲线 ───────────────────────────────────────────────────────────

def plot_training_curves(history, model_name="AE"):
    """绘制训练集和验证集 loss 曲线。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(history["train_loss"], label="Train Loss", linewidth=1.5)
    ax.plot(history["val_loss"], label="Validation Loss", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title(f"{model_name} Training Curve")
    ax.legend()
    save_or_show(fig, f"training_curve_{model_name}.png")


# ── PR 和 ROC 曲线 ─────────────────────────────────────────────────────

def plot_pr_curve(scores_dict, y_true):
    """在同一张图上绘制多个方法的 PR 曲线。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores in scores_dict.items():
        precision, recall, _ = precision_recall_curve(y_true, scores)
        ap = auc(recall, precision)  # 用曲线面积近似 AP
        ax.plot(recall, precision, linewidth=1.5, label=f"{name} (AP={ap:.3f})")
    no_skill = y_true.sum() / len(y_true)
    ax.axhline(no_skill, linestyle="--", color="gray", alpha=0.5, label=f"Random ({no_skill:.4f})")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title("Precision-Recall Curves")
    ax.legend(fontsize=8)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.05)
    save_or_show(fig, "pr_curves.png")


def plot_roc_curve(scores_dict, y_true):
    """绘制多个方法的 ROC 曲线。"""
    fig, ax = plt.subplots(figsize=(7, 6))
    for name, scores in scores_dict.items():
        fpr, tpr, _ = roc_curve(y_true, scores)
        auc_val = auc(fpr, tpr)
        ax.plot(fpr, tpr, linewidth=1.5, label=f"{name} (AUC={auc_val:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curves")
    ax.legend(fontsize=8)
    save_or_show(fig, "roc_curves.png")


# ── 混淆矩阵 ───────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true, y_pred, method_name):
    """绘制单个模型的混淆矩阵。"""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {method_name}")
    save_or_show(fig, f"confusion_{method_name}.png")


# ── 重构误差分布 ───────────────────────────────────────────────────────

def plot_recon_error_distribution(errors_normal, errors_fraud, method_name="AE"):
    """绘制正常交易和欺诈交易的重构误差直方图。"""
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(errors_normal, bins=100, alpha=0.6, label="Normal", density=True, color="steelblue")
    ax.hist(errors_fraud, bins=100, alpha=0.6, label="Fraud", density=True, color="crimson")
    ax.set_xlabel("Reconstruction Error")
    ax.set_ylabel("Density")
    ax.set_title(f"Reconstruction Error Distribution — {method_name}")
    ax.legend()
    save_or_show(fig, f"recon_error_{method_name}.png")


# ── latent 空间 ───────────────────────────────────────────────────────

def plot_latent_space(z, y, method_name="AE", use_tsne=True, max_samples=5000):
    """用 t-SNE 或原始二维坐标可视化 latent 空间。
    大数据集会先抽样，避免 t-SNE 占用太多内存。"""
    # t-SNE 前先抽样
    if use_tsne and len(z) > max_samples:
        rng = np.random.RandomState(config.RANDOM_STATE)
        indices = rng.choice(len(z), max_samples, replace=False)
        z_sample = z[indices]
        y_sample = y[indices]
    else:
        z_sample = z
        y_sample = y

    if use_tsne and z_sample.shape[1] > 2:
        n_perplexity = min(30, max(5, len(z_sample) // 10))
        z_2d = TSNE(n_components=2, random_state=config.RANDOM_STATE,
                     perplexity=n_perplexity, max_iter=500).fit_transform(z_sample)
    else:
        z_2d = z_sample

    fig, ax = plt.subplots(figsize=(7, 6))
    scatter = ax.scatter(z_2d[:, 0], z_2d[:, 1], c=y_sample, cmap="coolwarm",
                         alpha=0.5, s=3, rasterized=True)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")
    ax.set_title(f"{method_name} Latent Space (colored by class)")
    legend = ax.legend(*scatter.legend_elements(), title="Class", fontsize=8)
    ax.add_artist(legend)
    save_or_show(fig, f"latent_{method_name}.png")


# ── 指标对比柱状图 ─────────────────────────────────────────────────────

def plot_metrics_comparison(all_metrics):
    """绘制 AUPRC、AUC-ROC 和 F1 的分组柱状图。"""
    methods = [m["method"] for m in all_metrics]
    auprc_vals = [m["auprc"] for m in all_metrics]
    auc_vals = [m["auc_roc"] for m in all_metrics]
    f1_vals = [m["f1"] for m in all_metrics]

    x = np.arange(len(methods))
    width = 0.25

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(x - width, auprc_vals, width, label="AUPRC", color="steelblue")
    ax.bar(x, auc_vals, width, label="AUC-ROC", color="coral")
    ax.bar(x + width, f1_vals, width, label="F1-Score", color="seagreen")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=15, ha="right", fontsize=8)
    ax.set_ylabel("Score")
    ax.set_title("Model Comparison")
    ax.legend()
    ax.set_ylim(0, 1.05)
    save_or_show(fig, "metrics_comparison.png")
