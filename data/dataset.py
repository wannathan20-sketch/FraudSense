"""
信用卡欺诈检测的数据读取、预处理和切分工具。
切分时使用分层抽样，避免极少数欺诈样本在某个集合里消失。
"""
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, RobustScaler

import config


def load_raw(path: str = None) -> pd.DataFrame:
    """读取原始 CSV，返回 DataFrame。"""
    path = path or config.DATA_PATH
    df = pd.read_csv(path)
    return df


def preprocess(df: pd.DataFrame, scaler_type: str = "standard"):
    """
    对全量特征做缩放，返回特征矩阵 X、标签 y 和 scaler。
    这个函数保留给 notebook 兼容；正式 pipeline 使用 split_and_preprocess，
    先切分再缩放，避免数据泄漏。
    """
    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    if scaler_type == "robust":
        scaler = RobustScaler()
    else:
        scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)
    return X_scaled, y, scaler


def _make_scaler(scaler_type: str):
    if scaler_type == "robust":
        return RobustScaler()
    return StandardScaler()


def train_val_test_split(X, y):
    """
    分层切成 train / val / test。
    先切出 20% test，再从剩余部分切出 20% val，整体比例约为 64/16/20。
    """
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=config.VAL_SIZE,
        random_state=config.RANDOM_STATE, stratify=y_temp
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def split_and_preprocess(df: pd.DataFrame, scaler_type: str = "standard"):
    """
    先切分，再只用训练集 fit scaler。
    验证集和测试集只 transform，避免把它们的统计信息泄漏进训练流程。
    """
    X = df.drop("Class", axis=1).values.astype(np.float32)
    y = df["Class"].values.astype(np.float32)

    X_train, X_val, X_test, y_train, y_val, y_test = train_val_test_split(X, y)

    scaler = _make_scaler(scaler_type)
    X_train = scaler.fit_transform(X_train).astype(np.float32, copy=False)
    X_val = scaler.transform(X_val).astype(np.float32, copy=False)
    X_test = scaler.transform(X_test).astype(np.float32, copy=False)

    return X_train, X_val, X_test, y_train, y_val, y_test, scaler


def split_normal_anomaly(X, y):
    """返回正常交易样本，用于无监督/半监督模型训练。"""
    X_normal = X[y == 0]
    return X_normal


def to_tensor(X, dtype=torch.float32):
    return torch.tensor(X, dtype=dtype)


def get_dataloader(X, batch_size=None, shuffle=True):
    """把 numpy 特征矩阵包装成 PyTorch DataLoader。"""
    batch_size = batch_size or config.AE_BATCH_SIZE
    ds = TensorDataset(to_tensor(X))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True)


def get_cv_splits(X, y):
    """生成分层 K 折切分。"""
    skf = StratifiedKFold(
        n_splits=config.N_FOLDS, shuffle=True, random_state=config.RANDOM_STATE
    )
    for train_idx, val_idx in skf.split(X, y):
        yield train_idx, val_idx
