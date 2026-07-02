"""
Autoencoder 和 VAE 的训练逻辑。
训练数据应只包含正常交易，并使用 early stopping 控制过拟合。
"""
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from tqdm import tqdm

import config


def model_device(model):
    """返回模型参数所在的设备。"""
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")


def move_batch_to_model_device(batch, model):
    """从 DataLoader batch 中取出特征，并移动到模型所在设备。"""
    x = batch[0] if isinstance(batch, (tuple, list)) else batch
    return x.to(model_device(model), non_blocking=True)


class EarlyStopping:
    """带 patience 的 early stopping，并保存最佳权重。"""
    def __init__(self, patience=15, min_delta=1e-6):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.best_state = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_state = copy.deepcopy(model.state_dict())
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        return self.early_stop


def train_autoencoder(model, train_loader, val_loader, epochs=None, lr=None, patience=None):
    """
    使用 MSE 重构损失训练 Autoencoder。
    loader 中应只放正常交易样本。
    """
    epochs = epochs or config.AE_EPOCHS
    lr = lr or config.AE_LR
    patience = patience or config.AE_PATIENCE

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": []}
    progress = tqdm(range(epochs), desc="训练 AE", unit="epoch")

    for epoch in progress:
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        for batch in train_loader:
            x = move_batch_to_model_device(batch, model)
            _, x_recon = model(x)
            loss = criterion(x_recon, x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        history["train_loss"].append(train_loss)

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = move_batch_to_model_device(batch, model)
                _, x_recon = model(x)
                val_loss += criterion(x_recon, x).item()
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        scheduler.step()
        progress.set_postfix({"train": f"{train_loss:.4f}", "val": f"{val_loss:.4f}"})

        if early_stop(val_loss, model):
            progress.set_description("AE 提前停止")
            break

    # 恢复验证集上最好的权重
    if early_stop.best_state:
        model.load_state_dict(early_stop.best_state)
    return model, history


def train_vae(model, train_loader, val_loader, epochs=None, lr=None, patience=None, beta=None):
    """
    使用 MSE 重构损失 + KL 散度训练 VAE。
    """
    epochs = epochs or config.VAE_EPOCHS
    lr = lr or config.VAE_LR
    patience = patience or config.VAE_PATIENCE
    beta = beta or config.VAE_BETA

    mse = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=20, T_mult=2, eta_min=1e-6)
    early_stop = EarlyStopping(patience=patience)

    history = {"train_loss": [], "val_loss": [], "train_recon": [], "train_kl": []}
    progress = tqdm(range(epochs), desc="训练 VAE", unit="epoch")

    for epoch in progress:
        # ── 训练 ──
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_kl = 0.0
        for batch in train_loader:
            x = move_batch_to_model_device(batch, model)
            _, x_recon, mu, logvar = model(x)

            recon_loss = mse(x_recon, x)
            kl_loss = model.kl_divergence(mu, logvar).mean()
            loss = recon_loss + beta * kl_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_kl += kl_loss.item()

        n = len(train_loader)
        history["train_loss"].append(train_loss / n)
        history["train_recon"].append(train_recon / n)
        history["train_kl"].append(train_kl / n)

        # ── 验证 ──
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                x = move_batch_to_model_device(batch, model)
                _, x_recon, mu, logvar = model(x)
                recon_loss = mse(x_recon, x)
                kl_loss = model.kl_divergence(mu, logvar).mean()
                val_loss += (recon_loss + beta * kl_loss).item()
        val_loss /= len(val_loader)
        history["val_loss"].append(val_loss)

        scheduler.step()
        progress.set_postfix({"loss": f"{train_loss/n:.4f}", "val": f"{val_loss:.4f}"})

        if early_stop(val_loss, model):
            progress.set_description("VAE 提前停止")
            break

    if early_stop.best_state:
        model.load_state_dict(early_stop.best_state)
    return model, history
