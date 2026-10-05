"""Monte Carlo Dropout Inference Engine for Epistemic & Aleatoric Uncertainty Estimation.

This module keeps dropout active during inference across T stochastic passes to derive:
1. Predictive Mean Probability: mu_p = 1/T sum(p_t)
2. Epistemic Uncertainty (Predictive Variance): sigma^2 = 1/T sum((p_t - mu_p)^2)
3. Predictive Total Shannon Entropy: H(mu_p)
4. Expected Entropy (Aleatoric) & Mutual Information (Epistemic).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch

from dl.src.config import PipelineConfig, get_default_checkpoint_dir, get_default_output_dir, set_seeds
from dl.src.gnn import GraphSAGEModel


def enable_dropout_only(model: torch.nn.Module) -> None:
    """Put model in eval mode but activate all Dropout layers for MC sampling."""
    model.eval()
    for m in model.modules():
        if isinstance(m, torch.nn.Dropout):
            m.train()


def run_mc_dropout_sampling(
    model: GraphSAGEModel,
    x: torch.Tensor,
    edge_index: torch.Tensor,
    mask: torch.Tensor,
    num_samples: int = 30,
) -> Dict[str, np.ndarray]:
    """Execute T stochastic forward passes and compute uncertainty statistics."""
    enable_dropout_only(model)
    stochastic_probs = []

    print(f"[*] Running {num_samples} Monte Carlo Dropout passes with active dropout...")
    with torch.no_grad():
        for t in range(num_samples):
            logits = model(x, edge_index)
            probs = torch.sigmoid(logits[mask]).cpu().numpy()
            stochastic_probs.append(probs)

    # Shape: (T, N_masked)
    prob_matrix = np.stack(stochastic_probs, axis=0)

    # 1. Predictive Mean Probability
    mean_prob = np.mean(prob_matrix, axis=0)

    # 2. Epistemic Uncertainty (Predictive Variance)
    variance = np.var(prob_matrix, axis=0)

    # 3. Shannon Entropy
    eps = 1e-7
    p_clamped = np.clip(mean_prob, eps, 1.0 - eps)
    total_entropy = -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))

    # 4. Aleatoric vs Epistemic Decomposition
    probs_clamped = np.clip(prob_matrix, eps, 1.0 - eps)
    sample_entropies = -(probs_clamped * np.log2(probs_clamped) + (1.0 - probs_clamped) * np.log2(1.0 - probs_clamped))
    aleatoric_entropy = np.mean(sample_entropies, axis=0)
    mutual_info = np.maximum(0.0, total_entropy - aleatoric_entropy)

    return {
        "mean_prob": mean_prob,
        "variance": variance,
        "total_entropy": total_entropy,
        "aleatoric_entropy": aleatoric_entropy,
        "mutual_info": mutual_info,
        "raw_samples": prob_matrix,
    }


def compute_mc_uncertainties(config: PipelineConfig) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Compute and save MC Dropout uncertainty profiles on validation and test splits."""
    config.ensure_dirs()
    set_seeds(config.seed)
    device = torch.device(config.device)

    # Load preprocessed arrays and graph artifacts
    arrays = np.load(config.out_dir / "processed_arrays.npz")
    X_train, y_train = arrays["X_train"], arrays["y_train"]
    X_val, y_val = arrays["X_val"], arrays["y_val"]
    X_test, y_test = arrays["X_test"], arrays["y_test"]

    n_train, n_val, n_test = len(X_train), len(X_val), len(X_test)
    n_total = n_train + n_val + n_test

    X_all = np.vstack([X_train, X_val, X_test])
    graph_npz = config.out_dir / "graph_features.npz"
    if graph_npz.exists():
        gf = np.load(graph_npz)["graph_features"]
        X_all = np.hstack([X_all, gf])

    ae_npz = config.out_dir / "autoencoder_features.npz"
    if ae_npz.exists():
        ae_data = np.load(ae_npz)
        ae_err = np.vstack([ae_data["train_errors"], ae_data["val_errors"], ae_data["test_errors"]])
        ae_lat = np.vstack([ae_data["train_latents"], ae_data["val_latents"], ae_data["test_latents"]])
        X_all = np.hstack([X_all, ae_err, ae_lat])

    X_all_t = torch.tensor(X_all, dtype=torch.float32, device=device)
    edge_index = torch.load(config.out_dir / "edge_index.pt", map_location=device)

    # Masks
    val_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    val_mask[n_train:n_train + n_val] = True

    test_mask = torch.zeros(n_total, dtype=torch.bool, device=device)
    test_mask[n_train + n_val:] = True

    # Instantiate model & load checkpoint
    model = GraphSAGEModel(
        in_channels=X_all.shape[1],
        hidden_dim=config.gnn_hidden_dim,
        num_layers=config.gnn_num_layers,
        dropout=config.mc_dropout_rate,
    ).to(device)

    ckpt_path = config.checkpoint_dir / "gnn_ae.pt"
    if not ckpt_path.exists():
        ckpt_path = config.checkpoint_dir / "gnn.pt"

    model.load_state_dict(torch.load(ckpt_path, map_location=device))

    # MC Dropout inference
    val_metrics = run_mc_dropout_sampling(
        model, X_all_t, edge_index, val_mask, num_samples=config.mc_samples
    )
    test_metrics = run_mc_dropout_sampling(
        model, X_all_t, edge_index, test_mask, num_samples=config.mc_samples
    )

    # Save uncertainty artifacts
    np.savez_compressed(config.out_dir / "mc_dropout_val.npz", **val_metrics)
    np.savez_compressed(config.out_dir / "mc_dropout_test.npz", **test_metrics)

    print(f"[+] Saved MC Dropout uncertainty artifacts (T={config.mc_samples}) to {config.out_dir}")
    print(f"  • Test Mean Uncertainty (Variance): {np.mean(test_metrics['variance']):.6f} (Max: {np.max(test_metrics['variance']):.6f})")
    print(f"  • Test Mean Total Entropy: {np.mean(test_metrics['total_entropy']):.4f} bits")

    return val_metrics, test_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run MC Dropout uncertainty quantification.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--ckpt_dir", type=Path, default=get_default_checkpoint_dir(), help="Checkpoint directory")
    parser.add_argument("--samples", type=int, default=30, help="Number of Monte Carlo passes")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, checkpoint_dir=args.ckpt_dir, mc_samples=args.samples, seed=args.seed)
    compute_mc_uncertainties(cfg)
