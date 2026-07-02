"""
监督式表格模型的构造函数。
SMOTE 已经平衡了训练集，所以这里不再叠加 class weight。
"""
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

import config


def make_xgb_classifier(n_estimators=None):
    return XGBClassifier(
        n_estimators=n_estimators or config.XGB_N_ESTIMATORS,
        eval_metric="aucpr",
        random_state=config.RANDOM_STATE,
        verbosity=0,
        n_jobs=1,
    )


def make_lgb_classifier(n_estimators=None):
    return LGBMClassifier(
        n_estimators=n_estimators or config.LGB_N_ESTIMATORS,
        random_state=config.RANDOM_STATE,
        verbose=-1,
        n_jobs=1,
    )
