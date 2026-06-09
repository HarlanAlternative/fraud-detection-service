"""Autoencoder base learner — a deep unsupervised reconstruction signal.

A small dense autoencoder is trained to reconstruct **legitimate** transactions. Fraudulent
transactions, being out-of-distribution, reconstruct poorly, so the per-row reconstruction
error is an anomaly score. Kept intentionally simple (a plain dense AE, not a VAE) and
CPU-friendly — the goal is a credible "I can do unsupervised deep learning" signal.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from fraud.models.preprocess import build_preprocessor


def _to_dense(arr) -> np.ndarray:
    return arr.toarray() if hasattr(arr, "toarray") else np.asarray(arr)


class _DenseAE(nn.Module):
    def __init__(self, n_in: int, hidden: int = 32, latent: int = 8):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_in, hidden), nn.ReLU(),
            nn.Linear(hidden, latent), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, n_in),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


class AEScorer:
    """Reconstruction-error anomaly scorer with a stable 0-1 output."""

    def __init__(self, hidden: int = 32, latent: int = 8, epochs: int = 30, lr: float = 1e-3,
                 batch_size: int = 512, seed: int = 42):
        self.pre = build_preprocessor(scale=True)
        self.hidden, self.latent = hidden, latent
        self.epochs, self.lr, self.batch_size, self.seed = epochs, lr, batch_size, seed
        self.net: _DenseAE | None = None
        self._lo: float = 0.0
        self._hi: float = 1.0

    def _recon_error(self, Z: np.ndarray) -> np.ndarray:
        self.net.eval()
        with torch.no_grad():
            t = torch.from_numpy(Z.astype(np.float32))
            out = self.net(t)
            return ((out - t) ** 2).mean(dim=1).numpy()

    def fit(self, X: pd.DataFrame, y=None) -> AEScorer:
        torch.manual_seed(self.seed)
        Z_all = _to_dense(self.pre.fit_transform(X)).astype(np.float32)

        # Train on legitimate rows only ("learn normal"); fall back to all if no labels.
        if y is not None:
            mask = np.asarray(y) == 0
            Z_train = Z_all[mask] if mask.any() else Z_all
        else:
            Z_train = Z_all

        self.net = _DenseAE(Z_all.shape[1], self.hidden, self.latent)
        opt = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        loss_fn = nn.MSELoss()
        data = torch.from_numpy(Z_train)
        n = data.shape[0]

        self.net.train()
        for _ in range(self.epochs):
            perm = torch.randperm(n)
            for i in range(0, n, self.batch_size):
                batch = data[perm[i : i + self.batch_size]]
                opt.zero_grad()
                loss = loss_fn(self.net(batch), batch)
                loss.backward()
                opt.step()

        # Calibrate 0-1 range on the full training fold's reconstruction errors.
        err = self._recon_error(Z_all)
        self._lo, self._hi = np.percentile(err, [1, 99])
        if self._hi <= self._lo:
            self._hi = self._lo + 1e-9
        return self

    def score(self, X: pd.DataFrame) -> np.ndarray:
        """Return reconstruction-error anomaly scores in [0, 1] (1 = most anomalous)."""
        Z = _to_dense(self.pre.transform(X)).astype(np.float32)
        err = self._recon_error(Z)
        return np.clip((err - self._lo) / (self._hi - self._lo), 0.0, 1.0)
