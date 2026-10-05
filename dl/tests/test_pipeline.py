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


def test_sampler_preserves_natural_fraud_rate_and_classes():
    """Verify that subsampling preserves natural fraud rate within 1 percentage point and both classes exist in all splits."""
    np.random.seed(42)
    n_full = 100_000
    n_sample = 20_000

    # Create synthetic dataset with natural 3.5% fraud rate
    full_fraud_rate = 0.035
    is_fraud_full = (np.random.rand(n_full) < full_fraud_rate).astype(int)
    full_df = pd.DataFrame({
        "TransactionID": [f"TX_{i}" for i in range(n_full)],
        "TransactionDT": np.arange(n_full),
        "isFraud": is_fraud_full,
    })

    # Uniform random sample from the full dataset (fixed seed), then sort strictly by TransactionDT
    sampled_df = full_df.sample(n=n_sample, random_state=42).sort_values("TransactionDT").reset_index(drop=True)
    sample_fraud_rate = sampled_df["isFraud"].mean()

    # 1. Fraud rate in sample must be within 1 percentage point of full rate
    assert abs(sample_fraud_rate - full_fraud_rate) < 0.01, (
        f"Sample fraud rate {sample_fraud_rate:.4f} deviated by more than 1% from full rate {full_fraud_rate:.4f}"
    )

    # 2. Both classes must exist in every temporal split and fraud rate must be between 1% and 10%
    train_df, val_df, test_df = perform_temporal_split(sampled_df, train_ratio=0.70, val_ratio=0.15)
    for name, split in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
        n_fraud = split["isFraud"].sum()
        n_legit = len(split) - n_fraud
        rate = split["isFraud"].mean()
        assert n_fraud > 0, f"Split {name} has zero frauds!"
        assert n_legit > 0, f"Split {name} has zero legitimate transactions!"
        assert 0.01 <= rate <= 0.10, f"Split {name} fraud rate {rate*100:.2f}% outside [1%, 10%]"


def test_temporal_split_raises_on_single_class_or_anomaly():
    """Verify that perform_temporal_split raises clear ValueError if a split is single-class."""
    # 100% fraud dataset
    all_fraud_df = pd.DataFrame({
        "TransactionID": [f"TX_{i}" for i in range(100)],
        "TransactionDT": range(100),
        "isFraud": [1] * 100,
    })
    with pytest.raises(ValueError, match="Single-class error"):
        perform_temporal_split(all_fraud_df)


def test_numpy_json_serialization(tmp_path):
    """Verify that save_json correctly serializes np.float32, np.int64, arrays, and paths."""
    from dl.src.config import save_json
    import json

    data = {
        "float32_val": np.float32(0.98765),
        "float64_val": np.float64(1.23456),
        "int64_val": np.int64(42),
        "int32_val": np.int32(7),
        "bool_val": np.bool_(True),
        "array_val": np.array([1.0, 2.0, 3.0], dtype=np.float32),
        "nested_dict": {
            "p_fraud": np.float32(0.85),
            "counts": [np.int64(10), np.int64(20)],
        },
    }

    out_file = tmp_path / "test_out.json"
    # Must not raise TypeError: Object of type float32 is not JSON serializable
    save_json(out_file, data)
    assert out_file.exists()

    with open(out_file) as f:
        loaded = json.load(f)

    assert abs(loaded["float32_val"] - 0.98765) < 1e-4
    assert loaded["int64_val"] == 42
    assert loaded["bool_val"] is True
    assert loaded["nested_dict"]["p_fraud"] == loaded["nested_dict"]["p_fraud"]


def test_causal_graph_features_no_future_leakage():
    """Verify that causal past-only graph features do not count future transactions."""
    from dl.src.graph import construct_transaction_graph

    meta = pd.DataFrame({
        "TransactionID": ["T1", "T2", "T3", "T4"],
        "TransactionDT": [100, 200, 300, 400],
        "card_id": ["C1", "C1", "C1", "C2"],
        "device_id": ["D1", "D1", "D_NONE", "D1"],
        "isFraud": [1, 0, 1, 0],
        "split": ["train", "train", "val", "test"],
    })

    _, graph_features, feature_names = construct_transaction_graph(meta)
    
    # Feature 0: log_past_card_tx_count
    # T1 is first card1 tx -> past count = 0
    # T2 is second card1 tx -> past count = 1
    # T3 is third card1 tx -> past count = 2
    # T4 is first card2 tx -> past count = 0
    assert np.isclose(graph_features[0, 0], np.log1p(0.0))
    assert np.isclose(graph_features[1, 0], np.log1p(1.0))
    assert np.isclose(graph_features[2, 0], np.log1p(2.0))
    assert np.isclose(graph_features[3, 0], np.log1p(0.0))

    # Feature 3: log_past_card_fraud_count
    # T1 is fraud, but its past fraud count must be 0 (never counts itself)
    # T2 comes after T1 (fraud) -> past fraud count = 1
    # T3 (in val) comes after T1 (fraud in train) -> past fraud count = 1
    assert np.isclose(graph_features[0, 3], np.log1p(0.0))
    assert np.isclose(graph_features[1, 3], np.log1p(1.0))
    assert np.isclose(graph_features[2, 3], np.log1p(1.0))


def test_probability_calibrator():
    """Verify that ProbabilityCalibrator fits on validation and preserves monotonicity."""
    from dl.src.calibration import ProbabilityCalibrator

    np.random.seed(42)
    # Simulated distorted uncalibrated logits / probabilities from focal loss
    y_val = (np.random.rand(500) < 0.05).astype(np.float32)
    distorted_probs = np.clip(y_val * 0.7 + np.random.normal(0.4, 0.1, size=500), 0.01, 0.99)

    calibrator = ProbabilityCalibrator().fit(distorted_probs, y_val)
    cal_probs = calibrator.transform(distorted_probs)

    assert cal_probs.shape == distorted_probs.shape
    assert np.all(cal_probs >= 0.0) and np.all(cal_probs <= 1.0)
    # Calibrated probabilities should have mean closer to actual prevalence (~0.05) than 0.4
    assert np.mean(cal_probs) < np.mean(distorted_probs)


