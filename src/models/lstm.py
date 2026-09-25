"""LSTM (PyTorch) dengan output vektor multi-langkah.

* Input: jendela ``window`` hari berisi log harga ter-skala Min-Max (scaler di-fit
  **hanya** pada data train) + hari dalam minggu (sin/cos).
* Output: vektor 30 langkah berisi selisih log harga ter-skala terhadap origin,
  sehingga model tetap stabil saat harga keluar dari rentang historis.
* Regularisasi: dropout, early stopping pada validasi kronologis (tanpa shuffle
  antar waktu saat split). Residual validasi dipakai untuk interval empiris.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.models.base import Forecaster, empirical_bounds


class LSTMNet(nn.Module):
    """LSTM → dropout → linear ke vektor ``horizon`` langkah (level modul agar bisa di-pickle)."""

    def __init__(self, n_feat: int, hidden: int, layers: int, dropout: float, horizon: int):
        super().__init__()
        self.lstm = nn.LSTM(n_feat, hidden, num_layers=layers, batch_first=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.drop = nn.Dropout(dropout)
        self.head = nn.Linear(hidden, horizon)

    def forward(self, x):
        """Ambil state terakhir sekuens lalu proyeksikan ke semua langkah."""
        out, _ = self.lstm(x)
        return self.head(self.drop(out[:, -1, :]))


def make_windows(x: np.ndarray, window: int, horizon: int) -> tuple[np.ndarray, np.ndarray]:
    """Potong deret fitur ``(T, F)`` menjadi pasangan (input, target).

    Kolom 0 dari ``x`` adalah log harga ter-skala; target = x[t+k,0] − x[t,0].

    Returns:
        X bentuk ``(N, window, F)``, Y bentuk ``(N, horizon)``.
    """
    n = len(x) - window - horizon + 1
    if n <= 0:
        return np.empty((0, window, x.shape[1])), np.empty((0, horizon))
    idx = np.arange(window)[None, :] + np.arange(n)[:, None]
    X = x[idx]
    last = x[window - 1 + np.arange(n), 0]
    fut = np.arange(1, horizon + 1)[None, :] + (window - 1 + np.arange(n))[:, None]
    Y = x[fut, 0] - last[:, None]
    return X.astype(np.float32), Y.astype(np.float32)


class LSTMForecaster(Forecaster):
    """LSTM kecil (1–2 layer) untuk forecast multi-output."""

    name = "lstm"

    def __init__(self, window: int = 60, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2,
                 lr: float = 1e-3, batch_size: int = 64, max_epochs: int = 100, patience: int = 10,
                 val_ratio: float = 0.15, horizon: int = 30, seed: int = 42, n_threads: int = 2, **params):
        super().__init__(window=window, hidden_size=hidden_size, num_layers=num_layers, dropout=dropout,
                         lr=lr, batch_size=batch_size, max_epochs=max_epochs, patience=patience, **params)
        self.window, self.hidden_size, self.num_layers = int(window), int(hidden_size), int(num_layers)
        self.dropout, self.lr, self.batch_size = float(dropout), float(lr), int(batch_size)
        self.max_epochs, self.patience, self.val_ratio = int(max_epochs), int(patience), float(val_ratio)
        self.horizon, self.seed, self.n_threads = int(horizon), int(seed), int(n_threads)

    def _features(self, log_y: pd.Series) -> np.ndarray:
        scaled = (log_y.to_numpy() - self.min_) / (self.max_ - self.min_)
        dow = log_y.index.dayofweek.to_numpy()
        return np.column_stack([scaled, np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)])

    def _build(self):
        return LSTMNet(3, self.hidden_size, self.num_layers, self.dropout, self.horizon)

    def _fit(self, log_y: pd.Series) -> None:
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        torch.set_num_threads(self.n_threads)

        # Scaler Min-Max hanya dari data train (anti leakage).
        self.min_, self.max_ = float(log_y.min()), float(log_y.max())
        x = self._features(log_y)
        X, Y = make_windows(x, self.window, self.horizon)
        n_val = max(int(len(X) * self.val_ratio), 1)
        # Celah sebesar horizon antara train dan validasi agar target tidak tumpang tindih.
        X_tr, Y_tr = X[: -n_val - self.horizon], Y[: -n_val - self.horizon]
        X_va, Y_va = X[-n_val:], Y[-n_val:]

        net = self._build()
        opt = torch.optim.Adam(net.parameters(), lr=self.lr)
        loss_fn = torch.nn.MSELoss()
        Xt, Yt = torch.from_numpy(X_tr), torch.from_numpy(Y_tr)
        Xv, Yv = torch.from_numpy(X_va), torch.from_numpy(Y_va)
        gen = torch.Generator().manual_seed(self.seed)

        best, best_state, wait = np.inf, None, 0
        for _epoch in range(self.max_epochs):
            net.train()
            perm = torch.randperm(len(Xt), generator=gen)  # acak urutan batch, bukan urutan waktu
            for i in range(0, len(Xt), self.batch_size):
                b = perm[i:i + self.batch_size]
                opt.zero_grad()
                loss = loss_fn(net(Xt[b]), Yt[b])
                loss.backward()
                torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
                opt.step()
            net.eval()
            with torch.no_grad():
                val = loss_fn(net(Xv), Yv).item()
            if val < best - 1e-7:
                best, best_state, wait = val, copy.deepcopy(net.state_dict()), 0
            else:
                wait += 1
                if wait >= self.patience:
                    break
        net.load_state_dict(best_state)
        net.eval()
        self.net_ = net
        self.epochs_ = _epoch + 1
        with torch.no_grad():
            pred_va = net(Xv).numpy()
        scale = self.max_ - self.min_
        self.residuals_ = (Y_va - pred_va) * scale  # dalam satuan log harga
        self.history_ = log_y.iloc[-self.window:]

    def _predict(self, horizon, levels):
        x = self._features(self.history_)[None].astype(np.float32)
        with torch.no_grad():
            delta = self.net_(torch.from_numpy(x)).numpy()[0][:horizon]
        mean = self.history_.iloc[-1] + delta * (self.max_ - self.min_)
        return mean, empirical_bounds(mean, self.residuals_[:, :horizon], levels)
