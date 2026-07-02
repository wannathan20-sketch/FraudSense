"""
信用卡欺诈检测项目的集中配置。
路径、超参数和模型设置都放在这里，方便复现实验。
"""
import os

# ── 路径 ──────────────────────────────────────────────────────────────
DATA_PATH = os.path.join(os.path.dirname(__file__), "creditcard.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "checkpoints")

# ── 数据切分 ───────────────────────────────────────────────────────────
TEST_SIZE = 0.20
VAL_SIZE = 0.20          # 从 train+val 部分再切出 20% 作为验证集
RANDOM_STATE = 42
N_FOLDS = 5

# ── Autoencoder ───────────────────────────────────────────────────────
AE_HIDDEN_DIMS = [64, 32, 16]
AE_LATENT_DIM = 8        # 小瓶颈层用于学习正常交易的压缩表示
AE_DROPOUT = 0.2
AE_LR = 1e-3
AE_EPOCHS = 100
AE_PATIENCE = 15
AE_BATCH_SIZE = 256

# ── VAE ────────────────────────────────────────────────────────────────
VAE_HIDDEN_DIMS = [64, 32, 16]
VAE_LATENT_DIM = 2
VAE_DROPOUT = 0.2
VAE_LR = 1e-3
VAE_EPOCHS = 100
VAE_PATIENCE = 15
VAE_BATCH_SIZE = 256
VAE_BETA = 0.1            # KL 散度权重，保持较低以减少 posterior collapse 风险

# ── 传统机器学习模型 ───────────────────────────────────────────────────
RF_N_ESTIMATORS = 100
XGB_N_ESTIMATORS = 100
LGB_N_ESTIMATORS = 100

# ── 成本敏感评估 ───────────────────────────────────────────────────────
FRAUD_MISS_COST = 10      # 漏掉一笔欺诈相当于一次误报成本的 10 倍
