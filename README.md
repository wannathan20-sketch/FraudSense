# FraudSense — Credit Card Fraud Detection

Anomaly detection on the [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) (284,807 transactions, 0.17% fraud rate).

## Approach

Compare 5 methods across the unsupervised → supervised spectrum:

| Method | Type | AUPRC | AUC-ROC | Recall |
|--------|------|-------|---------|--------|
| Autoencoder | Semi-supervised | 0.509 | 0.950 | 0.786 |
| PCA Reconstruction | Unsupervised | 0.201 | 0.958 | 0.439 |
| Isolation Forest | Unsupervised | 0.138 | 0.953 | 0.378 |
| **XGBoost + SMOTE** | Supervised | **0.858** | **0.984** | 0.602 |
| LightGBM + SMOTE | Supervised | 0.728 | 0.983 | 0.674 |

**Key insight**: Accuracy is useless at 0.17% fraud rate. We use AUPRC, Precision@k, and Recall@k as primary metrics.

At 1% screening rate, XGBoost captures **89.8%** of all fraud.

## Project Structure

```
├── main.py              # One-command full pipeline
├── config.py            # All hyperparameters
├── data/
│   └── dataset.py       # Data loading, preprocessing, splits
├── models/
│   └── autoencoder.py   # AE + VAE architectures
├── train/
│   ├── train_ae.py      # Training + early stopping
│   └── evaluate.py      # Evaluation framework
├── utils/
│   ├── metrics.py       # AUPRC, Precision@k, Recall@k
│   └── visualization.py # All plots
└── notebooks/
    └── analysis.ipynb   # Presentation notebook
```

## Quick Start

```bash
# 1. Download dataset
kaggle datasets download mlg-ulb/creditcardfraud
unzip creditcardfraud.zip creditcard.csv -d .

# 2. Install dependencies
pip install torch scikit-learn xgboost lightgbm imbalanced-learn matplotlib seaborn tqdm

# 3. Run
python main.py
```

Results are saved to `outputs/`.

## Highlights

- **Semi-supervised**: Autoencoder trained exclusively on normal transactions — no fraud labels needed
- **No data leakage**: Scale after split; threshold tuned on validation set only
- **5 evaluation metrics**: AUPRC, AUC-ROC, F1, Precision@k, Recall@k, cost-sensitive scoring
- **Modular design**: Clean separation of config, data, models, training, and evaluation

## Data License

The [Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) is available under the Database Contents License (DbCL) v1.0.
