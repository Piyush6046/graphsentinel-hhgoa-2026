"""Graph construction module connecting transactions via shared cards and devices.

This module enforces:
1. Transaction-level graph node mapping across chronological splits.
2. Hub Degree Capping: Prevents O(N^2) memory explosion from mega-entities by capping
   maximum temporal neighborhood connections per card/device.
3. Graph topological feature extraction (degrees, temporal proximity, historical fraud ratio)
   fitted strictly on training labels to prevent target leakage.
4. Export of PyTorch edge indices and graph feature tables for GNN and GBT baselines.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch

from dl.src.config import PipelineConfig, get_default_output_dir, set_seeds


def cap_entity_clique(tx_indices: List[int], max_degree: int) -> List[Tuple[int, int]]:
    """Generate edges between transactions sharing an entity, capped by max_degree.
    
    If the number of transactions with this entity exceeds max_degree, each transaction
    is connected only to its nearest neighbors in time (chain/window graph) instead of
    a full O(K^2) clique.
    """
    k = len(tx_indices)
    if k <= 1:
        return []

    edges = []
    if k <= max_degree:
        # Full clique for small entities
        for i in range(k):
            for j in range(i + 1, k):
                edges.append((tx_indices[i], tx_indices[j]))
                edges.append((tx_indices[j], tx_indices[i]))
    else:
        # Bounded temporal window connections for high-degree hubs
        window_size = max(1, max_degree // 2)
        for i in range(k):
            # Connect to adjacent transactions in time
            left = max(0, i - window_size)
            right = min(k, i + window_size + 1)
            for j in range(left, right):
                if i != j:
                    edges.append((tx_indices[i], tx_indices[j]))
    return edges


def construct_transaction_graph(
    meta_df: pd.DataFrame, max_hub_degree: int = 50
) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """Build transaction graph edges and extract node-level topological features."""
    n_nodes = len(meta_df)
    meta_df = meta_df.reset_index(drop=True)
    meta_df["node_idx"] = np.arange(n_nodes)

    print(f"[*] Building graph across {n_nodes:,} transaction nodes (Hub degree cap: {max_hub_degree})...")

    all_edges: List[Tuple[int, int]] = []

    # 1. Edges via Shared Card ID
    card_groups = meta_df.groupby("card_id")["node_idx"].apply(list).to_dict()
    for card_id, indices in card_groups.items():
        if card_id and card_id != "C_NONE":
            all_edges.extend(cap_entity_clique(indices, max_hub_degree))

    # 2. Edges via Shared Device ID (ignoring missing/D_NONE devices)
    device_groups = meta_df.groupby("device_id")["node_idx"].apply(list).to_dict()
    for dev_id, indices in device_groups.items():
        if dev_id and dev_id != "D_NONE":
            all_edges.extend(cap_entity_clique(indices, max_hub_degree))

    # Deduplicate edges
    if all_edges:
        edge_set = set(all_edges)
        edge_src = [e[0] for e in edge_set]
        edge_dst = [e[1] for e in edge_set]
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
    else:
        # Self-loops if no edges formed
        edge_index = torch.tensor([list(range(n_nodes)), list(range(n_nodes))], dtype=torch.long)

    print(f"[*] Constructed graph with {edge_index.shape[1]:,} directed edges.")

    # 3. Compute Graph Topological Features
    # Node Degrees
    card_degrees = meta_df["card_id"].map(meta_df["card_id"].value_counts()).values.astype(np.float32)
    device_degrees = meta_df["device_id"].map(
        lambda d: meta_df["device_id"].value_counts().get(d, 1) if d != "D_NONE" else 1
    ).values.astype(np.float32)

    # Graph neighbor degree from edge_index
    node_degrees = np.zeros(n_nodes, dtype=np.float32)
    src_nodes, counts = np.unique(edge_index[0].numpy(), return_counts=True)
    node_degrees[src_nodes] = counts.astype(np.float32)

    # Historical fraud neighbor signal (strictly calculated using train labels only)
    # We assign 0 for val/test to prevent future label leakage
    train_mask = meta_df["split"] == "train"
    train_fraud_labels = meta_df["isFraud"].copy()
    train_fraud_labels[~train_mask] = 0.0  # Zero out val/test labels

    card_train_fraud_counts = meta_df[train_mask].groupby("card_id")["isFraud"].sum().to_dict()
    card_train_total_counts = meta_df[train_mask].groupby("card_id")["isFraud"].count().to_dict()

    historical_card_fraud_ratio = meta_df["card_id"].map(
        lambda c: (card_train_fraud_counts.get(c, 0.0) / card_train_total_counts.get(c, 1.0))
        if c in card_train_total_counts else 0.0
    ).values.astype(np.float32)

    graph_features = np.column_stack([
        np.log1p(card_degrees),
        np.log1p(device_degrees),
        np.log1p(node_degrees),
        historical_card_fraud_ratio,
    ]).astype(np.float32)

    graph_feature_names = [
        "log_card_degree", "log_device_degree", "log_graph_degree", "hist_card_fraud_ratio"
    ]

    return edge_index, graph_features, graph_feature_names


def build_graph(config: PipelineConfig) -> Tuple[torch.Tensor, np.ndarray, List[str]]:
    """Build and persist graph topology and features."""
    config.ensure_dirs()
    set_seeds(config.seed)

    # Load metadata splits
    train_meta = pd.read_parquet(config.out_dir / "train_meta.parquet")
    val_meta = pd.read_parquet(config.out_dir / "val_meta.parquet")
    test_meta = pd.read_parquet(config.out_dir / "test_meta.parquet")

    train_meta["split"] = "train"
    val_meta["split"] = "val"
    test_meta["split"] = "test"

    full_meta = pd.concat([train_meta, val_meta, test_meta], ignore_index=True)

    edge_index, graph_features, feature_names = construct_transaction_graph(
        full_meta, max_hub_degree=config.max_hub_degree
    )

    # Save edge index and graph features
    torch.save(edge_index, config.out_dir / "edge_index.pt")
    np.savez_compressed(
        config.out_dir / "graph_features.npz",
        graph_features=graph_features,
        feature_names=np.array(feature_names),
        n_train=len(train_meta),
        n_val=len(val_meta),
        n_test=len(test_meta),
    )

    print(f"[+] Saved edge index and graph features to {config.out_dir}")
    return edge_index, graph_features, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Construct graph and extract topological features.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output path for artifacts")
    parser.add_argument("--max_degree", type=int, default=50, help="Max hub degree cap")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, max_hub_degree=args.max_degree, seed=args.seed)
    build_graph(cfg)
