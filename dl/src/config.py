"""Central configuration and hyperparameter definitions for the DL fraud pipeline.

All paths, random seeds, model architectures, and training hyperparameters are
centralized here and can be overridden via CLI arguments or environment variables.
"""

from __future__ import annotations

import json
import os
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

import numpy as np
import torch


class NumpyJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder converting NumPy types to standard Python primitives."""

    def default(self, obj):
        if isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64, np.float16)):
            return float(obj)
        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        elif isinstance(obj, Path):
            return str(obj)
        return super().default(obj)


def save_json(path: Path | str, data: object, indent: int = 2) -> None:
    """Safely serialize and write data containing numpy types to a JSON file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyJSONEncoder, indent=indent)


def set_seeds(seed: int = 42) -> None:
    """Set random seeds across Python, NumPy, and PyTorch for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_default_data_dir() -> Path:
    """Resolve data directory based on environment (Kaggle vs local workspace)."""
    kaggle_dir = Path("/kaggle/input/ieee-fraud-detection")
    if kaggle_dir.exists():
        return kaggle_dir
    local_data = Path(__file__).resolve().parents[2] / "data"
    return local_data


def get_default_output_dir() -> Path:
    """Resolve output directory based on environment."""
    kaggle_out = Path("/kaggle/working/dl/results")
    if Path("/kaggle/working").exists():
        return kaggle_out
    return Path(__file__).resolve().parents[1] / "results"


def get_default_checkpoint_dir() -> Path:
    """Resolve checkpoint directory based on environment."""
    kaggle_ckpt = Path("/kaggle/working/dl/checkpoints")
    if Path("/kaggle/working").exists():
        return kaggle_ckpt
    return Path(__file__).resolve().parents[1] / "checkpoints"


@dataclass
class PipelineConfig:
    # Environment & Paths
    data_dir: Path = field(default_factory=get_default_data_dir)
    out_dir: Path = field(default_factory=get_default_output_dir)
    checkpoint_dir: Path = field(default_factory=get_default_checkpoint_dir)
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    seeds: List[int] = field(default_factory=lambda: [42, 123, 456])
    sample_size: int | None = None  # None for full dataset, integer for smoke-testing

    # Temporal Split Boundaries (approx 70% train, 15% val, 15% test)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Graph Construction Settings
    max_hub_degree: int = 50  # Cap maximum degree per card/device entity to prevent O(N^2) cliques
    include_email_edges: bool = True

    # Tabular Baseline Hyperparameters
    lgb_n_estimators: int = 400
    lgb_learning_rate: float = 0.05
    lgb_num_leaves: int = 63
    xgb_n_estimators: int = 400
    xgb_learning_rate: float = 0.05
    xgb_max_depth: int = 6

    # Tabular MLP Hyperparameters
    mlp_hidden_dims: List[int] = field(default_factory=lambda: [256, 128, 64])
    mlp_dropout: float = 0.25
    mlp_lr: float = 1e-3
    mlp_batch_size: int = 1024
    mlp_epochs: int = 30
    mlp_weight_decay: float = 1e-4

    # Autoencoder Hyperparameters (Trained strictly on legitimate train samples)
    ae_latent_dim: int = 16
    ae_hidden_dims: List[int] = field(default_factory=lambda: [128, 64, 32])
    ae_lr: float = 1e-3
    ae_batch_size: int = 1024
    ae_epochs: int = 25
    ae_weight_decay: float = 1e-5

    # GNN (GraphSAGE) Hyperparameters
    gnn_hidden_dim: int = 128
    gnn_num_layers: int = 2
    gnn_dropout: float = 0.30
    gnn_lr: float = 2e-3
    gnn_weight_decay: float = 5e-4
    gnn_epochs: int = 50
    gnn_patience: int = 10
    gnn_focal_gamma: float = 2.0
    gnn_focal_alpha: float = 0.25

    # Monte Carlo Dropout Inference
    mc_samples: int = 30
    mc_dropout_rate: float = 0.30

    # Next-Best-Action Decision Thresholds (Tuned strictly on Validation set)
    alert_budget_fraction: float = 0.01  # 1% top alerts
    cost_fp: float = 10.0  # Friction / manual investigation cost ($)
    cost_fn_multiplier: float = 1.0  # Fraud financial loss ($ exposure)
    evidence_request_cost: float = 2.0  # Cost of sending OTP/verification ($)

    def ensure_dirs(self) -> None:
        """Ensure all required output and checkpoint directories exist."""
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
