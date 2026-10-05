"""Graph Neural Network (GraphSAGE) architecture with Dual PyG / Pure-PyTorch Backend.

This module implements:
1. Seamless GraphSAGE with PyTorch Geometric or Pure-PyTorch Sparse fallback.
2. Dropout in all convolutional and classification layers for epistemic MC Dropout.
3. Focal Loss / Class-Weighted BCE loss for severe fraud imbalance.
4. Early stopping monitored strictly on validation PR-AUC.
5. Support for Graph Ablations (GNN vs. GNN + Autoencoder vs. MLP without graph).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import average_precision_score, roc_auc_score

from dl.src.config import PipelineConfig, get_default_checkpoint_dir, get_default_output_dir, set_seeds

# Try importing PyTorch Geometric; fallback to Pure-PyTorch Sparse SAGE if unavailable
try:
    from torch_geometric.nn import SAGEConv
    PYG_AVAILABLE = True
except ImportError:
    PYG_AVAILABLE = False


class SparseSAGEConv(nn.Module):
    """Pure PyTorch implementation of Mean-Aggregation GraphSAGE with sparse tensor ops."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.lin_self = nn.Linear(in_channels, out_channels, bias=False)
        self.lin_neigh = nn.Linear(in_channels, out_channels, bias=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin_self.weight)
        nn.init.xavier_uniform_(self.lin_neigh.weight)
        if self.lin_neigh.bias is not None:
            nn.init.zeros_(self.lin_neigh.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        n_nodes = x.size(0)
        device = x.device

        # Create sparse adjacency with row-normalization (mean neighbor aggregation)
        src, dst = edge_index[0], edge_index[1]
        deg = torch.bincount(dst, minlength=n_nodes).clamp(min=1).float()
        deg_inv = 1.0 / deg
        edge_weight = deg_inv[dst]

        # Sparse aggregation: neigh_x = A_norm @ x
        indices = torch.stack([dst, src], dim=0)
        adj_sparse = torch.sparse_coo_tensor(indices, edge_weight, (n_nodes, n_nodes), device=device)
        neigh_x = torch.sparse.mm(adj_sparse, x)

        out = self.lin_self(x) + self.lin_neigh(neigh_x)
        return out


class SAGEConvWrapper(nn.Module):
    """Unified wrapper choosing between PyG SAGEConv or pure-PyTorch SparseSAGEConv."""

    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        if PYG_AVAILABLE:
            self.conv = SAGEConv(in_channels, out_channels, aggr="mean")
        else:
            self.conv = SparseSAGEConv(in_channels, out_channels)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        return self.conv(x, edge_index)


class GraphSAGEModel(nn.Module):
    """Deep GraphSAGE network with LayerNorm, Dropout, and residual projections."""

    def __init__(
        self,
        in_channels: int,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.30,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.dropout_p = dropout

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        prev_dim = in_channels
        for _ in range(num_layers):
            self.convs.append(SAGEConvWrapper(prev_dim, hidden_dim))
            self.norms.append(nn.LayerNorm(hidden_dim))
            prev_dim = hidden_dim

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i in range(self.num_layers):
            h = self.convs[i](h, edge_index)
            h = self.norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout_p, training=self.training)

        logits = self.head(h).squeeze(-1)
        return logits


class BinaryFocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance in fraud detection."""

    def __init__(self, alpha: float = 0.25, gamma: float = 2.0, pos_weight: torch.Tensor | None = None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.pos_weight = pos_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=self.pos_weight, reduction="none"
        )
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1 - probs) * (1 - targets)
        focal_factor = (1.0 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            focal_factor = alpha_t * focal_factor

        return (focal_factor * bce_loss).mean()


def train_gnn_model(
    config: PipelineConfig, use_ae_features: bool = True, model_suffix: str = "gnn"
) -> Tuple[np.ndarray, np.ndarray, float, float, GraphSAGEModel]:
    """Train GraphSAGE on transaction graph with early stopping on Val PR-AUC."""
    config.ensure_dirs()
    set_seeds(config.seed)
    device = torch.device(config.device)

    print(f"[*] Training GraphSAGE (AE features: {use_ae_features}, Backend: {'PyG' if PYG_AVAILABLE else 'Pure-PyTorch Sparse'})...")

    # 1. Load data arrays
    arrays = np.load(config.out_dir / "processed_arrays.npz")
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    n_total = n_train + n_val + n_test

    X_all = np.vstack([X_train, X_val, X_test])
    y_all = np.concatenate([y_train, y_val, y_test])

    # 2. Append Graph Topological Features
    graph_npz = config.out_dir / "graph_features.npz"
    if graph_npz.exists():
        gf = np.load(graph_npz)["graph_features"]
        X_all = np.hstack([X_all, gf])

    # 3. Append Autoencoder Anomaly Score & Latent Features if requested
    if use_ae_features:
        ae_npz = config.out_dir / "autoencoder_features.npz"
        if ae_npz.exists():
            ae_data = np.load(ae_npz)
            ae_err = np.vstack([ae_data["train_errors"], ae_data["val_errors"], ae_data["test_errors"]])
            ae_lat = np.vstack([ae_data["train_latents"], ae_data["val_latents"], ae_data["test_latents"]])
            X_all = np.hstack([X_all, ae_err, ae_lat])

    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    y_all_t = torch.tensor(y_all, dtype=torch.float32, device=device)

    # 4. Load Edge Index
    edge_index = torch.load(config.out_dir / "edge_index.pt", map_location=device)

    # 5. Create Train/Val/Test boolean masks
    train_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    train_mask[:n_train] = True

    val_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    val_mask[n_train:n_train + n_val] = True

    test_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    test_mask[n_train + n_val:] = True

    # 6. Initialize Model & Loss
    model = GraphSAGEModel(
        in_channels=X_all.shape[1],
        hidden_dim=config.gnn_hidden_dim,
        num_layers=config.gnn_num_layers,
        dropout=config.gnn_dropout,
    ).to(device)

    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([max(1.0, float(n_neg) / max(1.0, float(n_pos)))], device=device)

    criterion = BinaryFocalLoss(
        alpha=config.gnn_focal_alpha,
        gamma=config.gnn_focal_gamma,
        pos_weight=pos_weight,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.gnn_lr, weight_decay=config.gnn_weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.gnn_epochs, eta_min=1e-5)

    best_val_pr_auc = 0.0
    best_state = None
    patience_counter = 0

    for epoch in range(1, config.gnn_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = model(X_all_t, edge_index)
        loss = criterion(logits[train_mask], y_all_t[train_mask])
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Validation Step
        model.eval()
        with torch.no_grad():
            val_logits = logits[val_mask]
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_pr_auc = float(average_precision_score(y_val, val_probs))

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config.gnn_patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Final Out-of-Sample Test Evaluation
    model.eval()
    with torch.no_grad():
        final_logits = model(X_all_t, edge_index)
        val_preds = torch.sigmoid(final_logits[val_mask]).cpu().numpy()
        test_preds = torch.sigmoid(final_logits[test_mask]).cpu().numpy()

    test_pr_auc = float(average_precision_score(y_test, test_preds))
    test_roc_auc = float(roc_auc_score(y_test, test_preds))

    name = f"GraphSAGE ({'Full GNN+AE' if use_ae_features else 'GNN without AE'})"
    print(f"  [+] {name} -> Val PR-AUC: {best_val_pr_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")

    # Save outputs & checkpoint
    ckpt_name = f"{model_suffix}.pt"
    torch.save(model.state_dict(), config.checkpoint_dir / ckpt_name)
    np.save(config.out_dir / f"preds_test_{model_suffix}.npy", test_preds)
    np.save(config.out_dir / f"preds_val_{model_suffix}.npy", val_preds)

    return val_preds, test_preds, test_pr_auc, test_roc_auc, model


def run_gnn_pipeline(config: PipelineConfig) -> Dict[str, Dict[str, float]]:
    """Run both standard GNN and GNN+AE models."""
    results = {}

    # GNN + Autoencoder (Full model)
    _, _, pr_full, roc_full, _ = train_gnn_model(config, use_ae_features=True, model_suffix="gnn_ae")
    results["GNN + AE"] = {"PR-AUC": pr_full, "ROC-AUC": roc_full}

    # GNN without AE (Ablation)
    _, _, pr_no_ae, roc_no_ae, _ = train_gnn_model(config, use_ae_features=False, model_suffix="gnn")
    results["GNN"] = {"PR-AUC": pr_no_ae, "ROC-AUC": roc_no_ae}

    with open(config.out_dir / "gnn_results.json", "w") as f:
        json.dump(results, f, indent=2)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train GraphSAGE neural model on transaction network.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--ckpt_dir", type=Path, default=get_default_checkpoint_dir(), help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, checkpoint_dir=args.ckpt_dir, seed=args.seed)
    run_gnn_pipeline(cfg)
