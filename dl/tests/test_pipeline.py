"""Unit tests for the Deep Learning fraud investigation pipeline.

Validates:
1. Strict temporal split ordering and zero future leakage.
2. Scaler and encoder parameter fitting isolation to the train split only.
3. MC Dropout stochastic variance properties (variance > 0 in stochastic mode, == 0 in eval mode).
4. Hub degree capping edge bounds.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from dl.src.config import PipelineConfig, set_seeds
from dl.src.data import perform_temporal_split, preprocess_tabular_features
from dl.src.gnn import GraphSAGEModel
from dl.src.graph import cap_entity_clique
from dl.src.mc_dropout import enable_dropout_only, run_mc_dropout_sampling


def test_temporal_split_strict_ordering():
    """Verify that temporal splitting enforces zero overlap between splits."""
    n = 1000
    df = pd.DataFrame({
        "TransactionID": [f"TX_{i}" for i in range(n)],
        "TransactionDT": np.arange(1000, 1000 + n),
        "TransactionAmt": np.random.exponential(50, size=n),
        "isFraud": (np.random.rand(n) < 0.05).astype(int),
        "card1": np.random.randint(1000, 2000, size=n),
        "card_id": [f"C_{i%50}" for i in range(n)],
        "device_id": [f"D_{i%30}" for i in range(n)],
    })

    train_df, val_df, test_df = perform_temporal_split(df, train_ratio=0.70, val_ratio=0.15)

    assert len(train_df) == 700
    assert len(val_df) == 150
    assert len(test_df) == 150

    assert train_df["TransactionDT"].max() < val_df["TransactionDT"].min()
    assert val_df["TransactionDT"].max() < test_df["TransactionDT"].min()


def test_preprocessor_fitting_isolation():
    """Verify that imputers and scalers are fitted strictly on train data."""
    np.random.seed(42)
    # Train with mean 10, Val with mean 50, Test with mean 100
    train_vals = np.random.normal(10.0, 1.0, size=100)
    val_vals = np.random.normal(50.0, 1.0, size=50)
    test_vals = np.random.normal(100.0, 1.0, size=50)

    train_df = pd.DataFrame({"TransactionAmt": train_vals, "isFraud": [0] * 100, "TransactionDT": range(100)})
    val_df = pd.DataFrame({"TransactionAmt": val_vals, "isFraud": [0] * 50, "TransactionDT": range(100, 150)})
    test_df = pd.DataFrame({"TransactionAmt": test_vals, "isFraud": [0] * 50, "TransactionDT": range(150, 200)})

    X_train, _, X_val, _, X_test, _, _ = preprocess_tabular_features(train_df, val_df, test_df)

    # Train mean should be near 0 after scaling, but Val and Test will have large positive scaled values
    assert np.abs(np.mean(X_train[:, 0])) < 0.1
    assert np.mean(X_val[:, 0]) > 20.0
    assert np.mean(X_test[:, 0]) > 50.0


def test_hub_degree_capping():
    """Verify that hub degree capping bounds the number of generated edges."""
    max_cap = 20
    # Mega entity with 500 transactions
    mega_entity_indices = list(range(500))

    edges = cap_entity_clique(mega_entity_indices, max_degree=max_cap)

    # Full clique would be 500 * 499 = 249,500 edges
    # Bounded window should be around 500 * 20 = 10,000 edges
    assert len(edges) < 20000
    assert len(edges) > 0


def test_mc_dropout_uncertainty_behavior():
    """Verify that MC Dropout produces strictly positive variance under stochastic passes."""
    set_seeds(42)
    in_dim = 16
    n_nodes = 20

    model = GraphSAGEModel(in_channels=in_dim, hidden_dim=32, num_layers=2, dropout=0.50)
    x = torch.randn(n_nodes, in_dim)
    edge_index = torch.tensor([list(range(n_nodes)), list(range(n_nodes))], dtype=torch.long)
    mask = torch.ones(n_nodes, dtype=torch.bool)

    # 1. Under standard eval mode without MC dropout, variance across passes should be 0.0
    model.eval()
    with torch.no_grad():
        p1 = torch.sigmoid(model(x, edge_index)).numpy()
        p2 = torch.sigmoid(model(x, edge_index)).numpy()
    np.testing.assert_allclose(p1, p2, atol=1e-6)

    # 2. Under stochastic MC dropout, variance must be strictly greater than 0
    mc_results = run_mc_dropout_sampling(model, x, edge_index, mask, num_samples=20)
    variance = mc_results["variance"]

    assert np.all(variance >= 0.0)
    assert np.mean(variance) > 0.0, "Expected positive epistemic variance under stochastic dropout!"
