# FraudSense — 信用卡欺诈异常检测

基于 [Credit Card Fraud Detection 数据集](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)（284,807 笔交易，欺诈率 0.17%），对比 5 种从无监督到有监督的异常检测方法。

## 方法对比

| 方法 | 类型 | AUPRC | AUC-ROC | Recall |
|------|------|-------|---------|--------|
| Autoencoder | 半监督 | 0.509 | 0.950 | 0.786 |
| PCA 重构误差 | 无监督 | 0.201 | 0.958 | 0.439 |
| Isolation Forest | 无监督 | 0.138 | 0.953 | 0.378 |
| **XGBoost + SMOTE** | 有监督 | **0.858** | **0.984** | 0.602 |
| LightGBM + SMOTE | 有监督 | 0.728 | 0.983 | 0.674 |

**核心观点**：0.17% 欺诈率下 Accuracy 毫无意义。本项目的评估体系以 AUPRC、Precision@k、Recall@k 为主指标。

仅审查 1% 交易时，XGBoost 可捕获 **89.8%** 的欺诈。

## 项目结构

```
├── main.py              # 一键运行完整 pipeline
├── config.py            # 所有超参集中管理
├── data/
│   └── dataset.py       # 数据加载、预处理、分层切分
├── models/
│   └── autoencoder.py   # Autoencoder + VAE 模型
├── train/
│   ├── train_ae.py      # 训练循环 + Early Stopping
│   └── evaluate.py      # 统一评估框架
├── utils/
│   ├── metrics.py       # AUPRC / Precision@k / Recall@k
│   └── visualization.py # 所有图表输出
└── notebooks/
    └── analysis.ipynb   # 展示用 notebook
```

## 快速开始

```bash
# 1. 下载数据集
kaggle datasets download mlg-ulb/creditcardfraud
unzip creditcardfraud.zip creditcard.csv -d .

# 2. 安装依赖
pip install torch scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn tqdm

# 3. 运行
python main.py
```

结果图表保存在 `outputs/` 目录。

## 技术要点

- **半监督异常检测**：Autoencoder 仅用正常交易训练，以重构误差为异常分数，无需欺诈标签
- **无数据泄漏**：先切分再标准化；阈值在验证集上独立搜索，不在测试集上调参
- **多维评估体系**：AUPRC、AUC-ROC、F1、Precision@k、Recall@k、Cost-sensitive
- **模块化架构**：config / data / models / train / utils 五层分离，训练模块统一封装早停与学习率调度

## 数据许可

[Credit Card Fraud Detection 数据集](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) 遵循 Database Contents License (DbCL) v1.0。
