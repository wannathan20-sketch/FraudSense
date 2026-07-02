"""
异常检测模型的统一评估工具。
同时支持重构误差类分数和传统机器学习模型输出的风险分数。
"""
import csv
import json
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    precision_recall_curve,
)

import config


def find_optimal_threshold(scores, y_true, metric="f1"):
    """
    根据验证集搜索异常分数阈值。
    y_true=1 表示欺诈；scores 越高，越应被视为异常。
    """
    thresholds = np.percentile(scores, np.linspace(50, 99.9, 200))
    best_thresh = thresholds[0]
    best_score = 0.0

    for t in thresholds:
        y_pred = (scores >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "precision":
            score = precision_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            raise ValueError(f"不支持的阈值优化指标: {metric}")
        if score > best_score:
            best_score = score
            best_thresh = t

    return best_thresh, best_score


def compute_metrics(scores, y_true, threshold=None, method_name=""):
    """
    计算一组常用评估指标。
    如果 threshold 为 None，会在当前数据上找最佳 F1 阈值；
    正式测试时应避免这样做，优先使用验证集选出的阈值。
    """
    if threshold is None:
        threshold, _ = find_optimal_threshold(scores, y_true)

    y_pred = (scores >= threshold).astype(int)

    metrics = {
        "method": method_name,
        "threshold": threshold,
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc_roc": roc_auc_score(y_true, scores),
        "auprc": average_precision_score(y_true, scores),
    }

    # 成本敏感口径：漏掉一笔欺诈的成本更高
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics["false_alarms"] = int(fp)
        metrics["missed_fraud"] = int(fn)
        metrics["total_cost"] = fp + config.FRAUD_MISS_COST * fn

    return metrics


def compute_metrics_from_validation_threshold(
    val_scores,
    y_val,
    test_scores,
    y_test,
    method_name="",
    metric="f1",
):
    """先在验证集上选阈值，再只在测试集上评估一次。"""
    threshold, _ = find_optimal_threshold(val_scores, y_val, metric=metric)
    return compute_metrics(
        test_scores,
        y_test,
        threshold=threshold,
        method_name=method_name,
    )


def evaluate_all_thresholds(scores, y_true):
    """计算 precision-recall 曲线需要的数据。"""
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    return precision, recall, thresholds


def format_metrics_table(all_metrics):
    """把多组指标格式化成终端表格。"""
    rows = []
    header = f"{'Method':<25} {'AUPRC':>8} {'AUC-ROC':>8} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Threshold':>10}"
    rows.append(header)
    rows.append("-" * len(header))
    for m in all_metrics:
        rows.append(
            f"{m['method']:<25} {m['auprc']:>8.4f} {m['auc_roc']:>8.4f} "
            f"{m['f1']:>8.4f} {m['precision']:>10.4f} {m['recall']:>8.4f} {m['threshold']:>10.4f}"
        )
    return "\n".join(rows)


def save_metrics(all_metrics, output_dir):
    """把模型指标保存为 CSV 和 JSON，方便报告和 README 使用。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "method",
        "auprc",
        "auc_roc",
        "f1",
        "precision",
        "recall",
        "threshold",
        "false_alarms",
        "missed_fraud",
        "total_cost",
    ]
    def normalize(value):
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
        return value

    rows = [
        {field: normalize(metrics.get(field, "")) for field in fieldnames}
        for metrics in all_metrics
    ]

    csv_path = output_dir / "metrics.csv"
    json_path = output_dir / "metrics.json"

    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    with json_path.open("w") as f:
        json.dump(rows, f, indent=2)

    return csv_path, json_path
