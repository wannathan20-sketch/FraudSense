"""
自定义评估指标工具。
"""
import numpy as np


def precision_at_k(y_true, scores, k_ratio=0.01):
    """
    top-k% Precision：风险最高的 k% 交易中，真正欺诈的比例。
    这个指标模拟人工审核资源有限的风控场景。
    """
    k = max(1, int(len(scores) * k_ratio))
    top_k_idx = np.argsort(scores)[-k:]
    y_top_k = y_true[top_k_idx]
    return y_top_k.sum() / k


def recall_at_k(y_true, scores, k_ratio=0.01):
    """top-k% Recall：风险最高的 k% 交易覆盖了多少真实欺诈。"""
    k = max(1, int(len(scores) * k_ratio))
    top_k_idx = np.argsort(scores)[-k:]
    y_top_k = y_true[top_k_idx]
    total_fraud = y_true.sum()
    return y_top_k.sum() / total_fraud if total_fraud > 0 else 0.0


def compute_all_at_k(y_true, scores, k_values=(0.005, 0.01, 0.02, 0.05)):
    """一次性计算多个 k 下的 Precision@k 和 Recall@k。"""
    result = {}
    for k in k_values:
        result[f"precision@{k:.1%}"] = precision_at_k(y_true, scores, k)
        result[f"recall@{k:.1%}"] = recall_at_k(y_true, scores, k)
    return result
