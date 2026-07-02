"""
用于异常检测的 Autoencoder 和 Variational Autoencoder。
模型用正常交易训练重构能力；如果欺诈交易偏离正常模式，重构误差通常会更高。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── 普通 Autoencoder ──────────────────────────────────────────────────

class Autoencoder(nn.Module):
    """
    对称结构的深层 autoencoder，包含 batchnorm 和 dropout。
    encoder 将输入压缩到 latent 表示，decoder 再从 latent 重构输入。
    """

    def __init__(self, input_dim, hidden_dims=(64, 32, 16), latent_dim=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ── 编码器 ──
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        encoder_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)

        # ── 解码器 ──
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
            ])
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        z = self.encoder(x)
        x_recon = self.decoder(z)
        return z, x_recon

    def reconstruction_error(self, x):
        """返回每条样本的 MSE 重构误差。"""
        _, x_recon = self.forward(x)
        return F.mse_loss(x_recon, x, reduction='none').mean(dim=1)


# ── Variational Autoencoder ───────────────────────────────────────────

class VAE(nn.Module):
    """
    带高斯 latent prior 的 VAE。
    这里用重构误差作为异常分数。
    """

    def __init__(self, input_dim, hidden_dims=(64, 32, 16), latent_dim=2, dropout=0.2):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        # ── 编码器 ──
        encoder_layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            encoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        self.encoder = nn.Sequential(*encoder_layers)

        # latent 分布参数
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(hidden_dims[-1], latent_dim)

        # ── 解码器 ──
        decoder_layers = []
        prev_dim = latent_dim
        for h in reversed(hidden_dims):
            decoder_layers.extend([
                nn.Linear(prev_dim, h),
                nn.ReLU(),
            ])
            prev_dim = h
        decoder_layers.append(nn.Linear(prev_dim, input_dim))
        self.decoder = nn.Sequential(*decoder_layers)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        z = self.reparameterize(mu, logvar)
        x_recon = self.decoder(z)
        return z, x_recon, mu, logvar

    def reconstruction_error(self, x, n_samples=1):
        """
        通过多次采样估计重构误差。
        误差越高，越可能是异常交易。
        """
        errors = []
        for _ in range(n_samples):
            _, x_recon, mu, logvar = self.forward(x)
            err = F.mse_loss(x_recon, x, reduction='none').mean(dim=1)
            errors.append(err)
        return torch.stack(errors, dim=0).mean(dim=0)

    def kl_divergence(self, mu, logvar):
        """计算 N(mu, sigma) 到标准正态 N(0, 1) 的 KL 散度。"""
        return -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
