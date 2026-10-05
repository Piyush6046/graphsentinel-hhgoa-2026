"""Baseline tabular models: LightGBM, XGBoost, Tabular MLP, and LightGBM + Graph Features.

All models are trained strictly on the train split, validated on the validation split,
and produce out-of-sample probability predictions on the temporal test set.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import lightgbm as lgb
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import average_precision_score, roc_auc_score
from torch.utils.data import DataLoader, TensorDataset
import xgboost as xgb

from dl.src.config import PipelineConfig, get_default_checkpoint_dir, get_default_output_dir, save_json, set_seeds


class TabularMLP(nn.Module):
    """Deep Tabular Multilayer Perceptron with Batch Normalization and Dropout."""

    def __init__(self, in_features: int, hidden_dims: list[int], dropout: float = 0.25):
        super().__init__()
        layers = []
        prev_dim = in_features
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim
        layers.append(nn.Linear(prev_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def validate_binary_targets(y: np.ndarray, split_name: str) -> None:
    """Ensure target array contains both positive and negative classes."""
    unique = np.unique(y)
    if len(unique) < 2:
        raise ValueError(
            f"Invalid target distribution in {split_name}: found only class(es) {unique.tolist()}. "
            f"Both positive (fraud=1) and negative (legitimate=0) classes are strictly required."
        )


def train_lightgbm_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    config: PipelineConfig,
    feature_prefix: str = "lgb",
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Train LightGBM gradient-boosted tree baseline."""
    validate_binary_targets(y_train, "LightGBM Train")
    validate_binary_targets(y_val, "LightGBM Val")
    validate_binary_targets(y_test, "LightGBM Test")

    print(f"[*] Training LightGBM baseline ({feature_prefix})...")

    # Pos weight to balance fraud classes
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = max(1.0, float(n_neg) / max(1.0, float(n_pos)))

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": "average_precision",
        "boosting_type": "gbdt",
        "learning_rate": config.lgb_learning_rate,
        "num_leaves": config.lgb_num_leaves,
        "scale_pos_weight": scale_pos_weight,
        "random_state": config.seed,
        "verbose": -1,
        "n_jobs": -1,
    }

    model = lgb.train(
        params,
        train_data,
        num_boost_round=config.lgb_n_estimators,
        valid_sets=[train_data, val_data],
        callbacks=[
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ],
    )

    val_preds = model.predict(X_val)
    test_preds = model.predict(X_test)

    val_pr_auc = float(average_precision_score(y_val, val_preds))
    test_pr_auc = float(average_precision_score(y_test, test_preds))
    test_roc_auc = float(roc_auc_score(y_test, test_preds))

    print(f"  [+] LightGBM ({feature_prefix}) -> Val PR-AUC: {val_pr_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
    return val_preds, test_preds, test_pr_auc, test_roc_auc


def train_xgboost_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """Train XGBoost gradient-boosted tree baseline."""
    validate_binary_targets(y_train, "XGBoost Train")
    validate_binary_targets(y_val, "XGBoost Val")
    validate_binary_targets(y_test, "XGBoost Test")

    print("[*] Training XGBoost baseline...")
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    scale_pos_weight = max(1.0, float(n_neg) / max(1.0, float(n_pos)))

    model = xgb.XGBClassifier(
        n_estimators=config.xgb_n_estimators,
        learning_rate=config.xgb_learning_rate,
        max_depth=config.xgb_max_depth,
        scale_pos_weight=scale_pos_weight,
        eval_metric="aucpr",
        random_state=config.seed,
        n_jobs=-1,
        early_stopping_rounds=30,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        verbose=False,
    )

    val_preds = model.predict_proba(X_val)[:, 1]
    test_preds = model.predict_proba(X_test)[:, 1]

    val_pr_auc = float(average_precision_score(y_val, val_preds))
    test_pr_auc = float(average_precision_score(y_test, test_preds))
    test_roc_auc = float(roc_auc_score(y_test, test_preds))

    print(f"  [+] XGBoost -> Val PR-AUC: {val_pr_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
    return val_preds, test_preds, test_pr_auc, test_roc_auc


def train_mlp_baseline(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, float, float, TabularMLP]:
    """Train PyTorch Tabular MLP baseline."""
    validate_binary_targets(y_train, "Tabular MLP Train")
    validate_binary_targets(y_val, "Tabular MLP Val")
    validate_binary_targets(y_test, "Tabular MLP Test")

    print("[*] Training Tabular MLP baseline...")
    device = torch.device(config.device)

    train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train, dtype=torch.float32))
    train_loader = DataLoader(train_dataset, batch_size=config.mlp_batch_size, shuffle=True)

    model = TabularMLP(
        in_features=X_train.shape[1],
        hidden_dims=config.mlp_hidden_dims,
        dropout=config.mlp_dropout,
    ).to(device)

    # Weighted BCE loss for class imbalance
    n_neg = (y_train == 0).sum()
    n_pos = (y_train == 1).sum()
    pos_weight = torch.tensor([max(1.0, float(n_neg) / max(1.0, float(n_pos)))], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=config.mlp_lr, weight_decay=config.mlp_weight_decay)

    best_val_pr_auc = 0.0
    best_state = None

    X_val_t = torch.tensor(X_val, dtype=torch.float32, device=device)
    X_test_t = torch.tensor(X_test, dtype=torch.float32, device=device)

    for epoch in range(config.mlp_epochs):
        model.train()
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()

        # Evaluate on Val
        model.eval()
        with torch.no_grad():
            val_logits = model(X_val_t)
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_pr_auc = float(average_precision_score(y_val, val_probs))

        if val_pr_auc > best_val_pr_auc:
            best_val_pr_auc = val_pr_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    with torch.no_grad():
        val_preds = torch.sigmoid(model(X_val_t)).cpu().numpy()
        test_preds = torch.sigmoid(model(X_test_t)).cpu().numpy()

    test_pr_auc = float(average_precision_score(y_test, test_preds))
    test_roc_auc = float(roc_auc_score(y_test, test_preds))

    print(f"  [+] Tabular MLP -> Val PR-AUC: {best_val_pr_auc:.4f} | Test PR-AUC: {test_pr_auc:.4f} | Test ROC-AUC: {test_roc_auc:.4f}")
    return val_preds, test_preds, test_pr_auc, test_roc_auc, model


def run_baselines(config: PipelineConfig) -> Dict[str, Dict[str, float]]:
    """Run all baseline models and export test predictions."""
    config.ensure_dirs()
    set_seeds(config.seed)

    # Load preprocessed arrays
    arrays = np.load(config.out_dir / "processed_arrays.npz")
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    # Load graph features if present for ablation
    graph_npz_path = config.out_dir / "graph_features.npz"
    if graph_npz_path.exists():
        gf_data = np.load(graph_npz_path)
        gf = gf_data["graph_features"]
        n_tr, n_v, n_te = int(gf_data["n_train"]), int(gf_data["n_val"]), int(gf_data["n_test"])
        gf_train = gf[:n_tr]
        gf_val = gf[n_tr:n_tr + n_v]
        gf_test = gf[n_tr + n_v:]

        X_train_graph = np.hstack([X_train, gf_train])
        X_val_graph = np.hstack([X_val, gf_val])
        X_test_graph = np.hstack([X_test, gf_test])
    else:
        X_train_graph, X_val_graph, X_test_graph = X_train, X_val, X_test

    results = {}

    # 1. LightGBM (Tabular)
    val_lgb, test_lgb, pr_lgb, roc_lgb = train_lightgbm_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test, config, "tabular"
    )
    results["LightGBM"] = {"PR-AUC": pr_lgb, "ROC-AUC": roc_lgb}
    np.save(config.out_dir / "preds_test_lgb.npy", test_lgb)
    np.save(config.out_dir / "preds_val_lgb.npy", val_lgb)

    # 2. XGBoost (Tabular)
    val_xgb, test_xgb, pr_xgb, roc_xgb = train_xgboost_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test, config
    )
    results["XGBoost"] = {"PR-AUC": pr_xgb, "ROC-AUC": roc_xgb}
    np.save(config.out_dir / "preds_test_xgb.npy", test_xgb)
    np.save(config.out_dir / "preds_val_xgb.npy", val_xgb)

    # 3. Tabular MLP (Neural)
    val_mlp, test_mlp, pr_mlp, roc_mlp, mlp_model = train_mlp_baseline(
        X_train, y_train, X_val, y_val, X_test, y_test, config
    )
    results["Tabular MLP"] = {"PR-AUC": pr_mlp, "ROC-AUC": roc_mlp}
    np.save(config.out_dir / "preds_test_mlp.npy", test_mlp)
    np.save(config.out_dir / "preds_val_mlp.npy", val_mlp)
    torch.save(mlp_model.state_dict(), config.checkpoint_dir / "mlp_baseline.pt")

    # 4. LightGBM + Graph Topological Features
    if graph_npz_path.exists():
        val_lgb_g, test_lgb_g, pr_lgb_g, roc_lgb_g = train_lightgbm_baseline(
            X_train_graph, y_train, X_val_graph, y_val, X_test_graph, y_test, config, "tabular+graph"
        )
        results["LightGBM + Graph"] = {"PR-AUC": pr_lgb_g, "ROC-AUC": roc_lgb_g}
        np.save(config.out_dir / "preds_test_lgb_graph.npy", test_lgb_g)
        np.save(config.out_dir / "preds_val_lgb_graph.npy", val_lgb_g)

    save_json(config.out_dir / "baseline_results.json", results)

    print(f"[+] Baseline evaluation complete. Artifacts saved to {config.out_dir}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train tabular baselines (LightGBM, XGBoost, MLP).")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--ckpt_dir", type=Path, default=get_default_checkpoint_dir(), help="Checkpoint directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, checkpoint_dir=args.ckpt_dir, seed=args.seed)
    run_baselines(cfg)
