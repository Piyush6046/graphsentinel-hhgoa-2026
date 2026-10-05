"""Unsupervised Tabular Autoencoder for Anomaly Detection & Representation Learning.

The Autoencoder is trained STRICTLY on legitimate (isFraud == 0) training samples.
Reconstruction error serves as an anomaly score and is exported alongside latent
bottleneck embeddings as auxiliary features for Graph Neural Networks.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from dl.src.config import PipelineConfig, get_default_checkpoint_dir, get_default_output_dir, save_json, set_seeds


class TabularAutoencoder(nn.Module):
    """Deep Symmetric Autoencoder for Tabular Representation & Anomaly Scoring."""

    def __init__(self, in_features: int, hidden_dims: list[int] = [128, 64, 32], latent_dim: int = 16):
        super().__init__()

        # Encoder
        enc_layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            enc_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        enc_layers.append(nn.Linear(prev_dim, latent_dim))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        prev_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            dec_layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
            ])
            prev_dim = h_dim
        dec_layers.append(nn.Linear(prev_dim, in_features))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        latent = self.encoder(x)
        reconstruction = self.decoder(latent)
        return reconstruction, latent


def train_autoencoder(config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray, TabularAutoencoder]:
    """Train Autoencoder on legitimate train transactions and compute anomaly scores."""
    config.ensure_dirs()
    set_seeds(config.seed)
    device = torch.device(config.device)

    arrays = np.load(config.out_dir / "processed_arrays.npz")
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    # Strict rule: Filter ONLY legitimate transactions for training
    legit_mask_train = (y_train == 0)
    X_train_legit = X_train[legit_mask_train]
    print(f"[*] Training Autoencoder on {len(X_train_legit):,} legitimate training samples...")

    train_dataset = TensorDataset(torch.tensor(X_train_legit, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.ae_batch_size, shuffle=True)

    in_dim = X_train.shape[1]
    model = TabularAutoencoder(
        in_features=in_dim,
        hidden_dims=config.ae_hidden_dims,
        latent_dim=config.ae_latent_dim,
    ).to(device)

    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.ae_lr, weight_decay=config.ae_weight_decay)

    # Evaluation tensors
    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)
    X_train_t = torch.tensor(X_train, dtype=torch.float32, device=device)

    best_val_loss = float("inf")
    best_state = None

    for epoch in range(config.ae_epochs):
        model.train()
        total_loss = 0.0
        for (batch_x,) in train_loader:
            batch_x = batch_x.to(device)
            optimizer.zero_grad()
            recon, _ = model(batch_x)
            loss = criterion(recon, batch_x)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch_x)

        avg_train_loss = total_loss / len(X_train_legit)

        # Validation on legitimate validation split
        model.eval()
        with torch.no_grad():
            val_recon, _ = model(X_val_t)
            # Evaluate reconstruction loss specifically on legitimate val samples
            val_legit_mask = (y_val == 0)
            val_loss = criterion(val_recon[val_legit_mask], X_val_t[val_legit_mask]).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    # Compute reconstruction errors (per-sample MSE across features)
    model.eval()
    with torch.no_grad():
        def compute_scores_and_latents(x_tensor: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
            recons, latents = [], []
            batch_size = 2048
            for i in range(0, len(x_tensor), batch_size):
                batch = x_tensor[i:i + batch_size]
                r, l = model(batch)
                mse_per_sample = torch.mean((r - batch) ** 2, dim=-1).cpu().numpy()
                recons.append(mse_per_sample)
                latents.append(l.cpu().numpy())
            return np.concatenate(recons), np.concatenate(latents)

        train_errors, train_latents = compute_scores_and_latents(X_train_t)
        val_errors, val_latents = compute_scores_and_latents(X_val_t)
        test_errors, test_latents = compute_scores_and_latents(X_test_t)

    # Evaluate unsupervised anomaly score on Test set
    test_pr_auc = float(average_precision_score(y_test, test_errors))
    test_roc_auc = float(roc_auc_score(y_test, test_errors))
    print(f"  [+] Unsupervised Autoencoder Anomaly Score -> Test PR-AUC: {test_pr_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")

    # Save checkpoint & features
    torch.save(model.state_dict(), config.checkpoint_dir / "autoencoder.pt")
    np.save(config.out_dir / "preds_test_ae.npy", test_errors)
    np.save(config.out_dir / "preds_val_ae.npy", val_errors)

    np.savez_compressed(
        config.out_dir / "autoencoder_features.npz",
        train_errors=train_errors[:, None],
        val_errors=val_errors[:, None],
        test_errors=test_errors[:, None],
        train_latents=train_latents,
        val_latents=val_latents,
        test_latents=test_latents,
    )

    save_json(config.out_dir / "autoencoder_results.json", {"Autoencoder": {"PR-AUC": test_pr_auc, "ROC-AUC": test_roc_auc}})

    print(f"[+] Autoencoder training complete. Checkpoint saved to {config.checkpoint_dir}")
    return train_errors, val_errors, test_errors, model


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train unsupervised autoencoder on legitimate transactions.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--ckpt_dir", type=Path, default=get_default_checkpoint_dir(), help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, checkpoint_dir=args.ckpt_dir, seed=args.seed)
    train_autoencoder(cfg)
