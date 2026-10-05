"""Data loading, composite key construction, temporal splitting, and feature preprocessing.

This module guarantees:
1. Strict temporal train/val/test splitting on TransactionDT.
2. All scalers, imputers, and categorical encoders are fitted ONLY on the training split.
3. No label or feature leakage from future validation or test splits.
4. Production of clean tabular features and entity keys for graph construction.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from dl.src.config import PipelineConfig, get_default_data_dir, get_default_output_dir, set_seeds


def find_file(directory: Path, candidates: List[str]) -> Path | None:
    """Find the first matching file from candidate names in directory."""
    for cand in candidates:
        p = directory / cand
        if p.exists():
            return p
        # Check case-insensitive
        for f in directory.glob("*"):
            if f.name.lower() == cand.lower():
                return f
    return None


def construct_composite_keys(df: pd.DataFrame) -> pd.DataFrame:
    """Construct stable composite card_id and device_id keys for relationship linking."""
    # Composite Card ID: card1-card6 + billing addr1/addr2
    card_cols = ["card1", "card2", "card3", "card4", "card5", "card6", "addr1", "addr2"]
    available_card_cols = [c for c in card_cols if c in df.columns]

    def make_card_key(row):
        parts = [str(row[c]) if pd.notna(row[c]) else "nan" for c in available_card_cols]
        raw_key = "_".join(parts)
        return "C_" + hashlib.md5(raw_key.encode()).hexdigest()[:12]

    df["card_id"] = df.apply(make_card_key, axis=1)

    # Composite Device ID: DeviceInfo + id_30 (OS) + id_31 (Browser) + id_33 (Screen)
    device_cols = ["DeviceInfo", "id_30", "id_31", "id_33", "DeviceType"]
    available_dev_cols = [c for c in device_cols if c in df.columns]

    if available_dev_cols:
        def make_device_key(row):
            parts = [str(row[c]).strip().lower() if pd.notna(row[c]) else "none" for c in available_dev_cols]
            if all(p == "none" for p in parts):
                return "D_NONE"
            raw_key = "_".join(parts)
            return "D_" + hashlib.md5(raw_key.encode()).hexdigest()[:12]

        df["device_id"] = df.apply(make_device_key, axis=1)
    else:
        df["device_id"] = "D_NONE"

    return df


def load_raw_ieee_data(data_dir: Path, sample_size: int | None = None) -> pd.DataFrame:
    """Load IEEE-CIS transaction and identity files, merge on TransactionID, and construct keys."""
    tx_file = find_file(data_dir, [
        "train_transaction.csv", "train_transaction.parquet", "transactions.csv", "transactions.parquet"
    ])
    id_file = find_file(data_dir, [
        "train_identity.csv", "train_identity.parquet", "identity.csv", "identity.parquet"
    ])

    if tx_file is None:
        raise FileNotFoundError(
            f"Could not find IEEE-CIS transaction file in '{data_dir}'.\n"
            f"Expected 'train_transaction.csv' or 'train_transaction.parquet'.\n"
            f"If running on Kaggle, set --data_dir /kaggle/input/ieee-fraud-detection/\n"
            f"If running locally, place raw IEEE-CIS files into '{data_dir}'."
        )

    print(f"[*] Loading transactions from {tx_file}...")
    if tx_file.suffix == ".parquet":
        tx_df = pd.read_parquet(tx_file)
    else:
        tx_df = pd.read_csv(tx_file, low_memory=False)

    if id_file is not None:
        print(f"[*] Loading identity signals from {id_file}...")
        if id_file.suffix == ".parquet":
            id_df = pd.read_parquet(id_file)
        else:
            id_df = pd.read_csv(id_file, low_memory=False)
        merged_df = tx_df.merge(id_df, on="TransactionID", how="left")
    else:
        print("[!] Warning: Identity file not found; proceeding with transaction signals only.")
        merged_df = tx_df

    # Verify or create isFraud label
    if "isFraud" not in merged_df.columns:
        raise KeyError(
            "Missing 'isFraud' target column in transaction table. "
            "Please ensure the official IEEE-CIS train dataset is provided."
        )

    # Sort strictly by TransactionDT for temporal fidelity
    merged_df = merged_df.sort_values("TransactionDT").reset_index(drop=True)

    if sample_size is not None and sample_size < len(merged_df):
        print(f"[*] Subsampling to {sample_size:,} time-ordered records (preserving all fraud cases)...")
        # Keep all fraud records in the temporal window, and sample legitimate rows to reach sample_size
        fraud_mask = merged_df["isFraud"] == 1
        legit_mask = ~fraud_mask
        n_fraud = fraud_mask.sum()
        n_legit_needed = max(0, sample_size - n_fraud)

        legit_sampled = merged_df[legit_mask].sample(n=min(n_legit_needed, legit_mask.sum()), random_state=42)
        merged_df = pd.concat([merged_df[fraud_mask], legit_sampled]).sort_values("TransactionDT").reset_index(drop=True)

    print(f"[*] Constructing composite card and device entity keys...")
    merged_df = construct_composite_keys(merged_df)
    return merged_df


def perform_temporal_split(
    df: pd.DataFrame, train_ratio: float = 0.70, val_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Perform a strict chronological split based on TransactionDT."""
    df = df.sort_values("TransactionDT").reset_index(drop=True)
    n = len(df)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_df = df.iloc[:train_end].copy().reset_index(drop=True)
    val_df = df.iloc[train_end:val_end].copy().reset_index(drop=True)
    test_df = df.iloc[val_end:].copy().reset_index(drop=True)

    def print_split_stats(name: str, split_df: pd.DataFrame):
        n_rows = len(split_df)
        n_fraud = int(split_df["isFraud"].sum())
        rate = (n_fraud / n_rows * 100) if n_rows > 0 else 0.0
        dt_min, dt_max = split_df["TransactionDT"].min(), split_df["TransactionDT"].max()
        print(f"  • {name:5s}: {n_rows:7,d} rows | Frauds: {n_fraud:5,d} ({rate:5.2f}%) | DT range: [{dt_min} -> {dt_max}]")

    print("[*] Temporal Split Summary:")
    print_split_stats("Train", train_df)
    print_split_stats("Val", val_df)
    print_split_stats("Test", test_df)

    # Sanity check: Ensure zero temporal overlap
    assert train_df["TransactionDT"].max() <= val_df["TransactionDT"].min(), "Temporal leakage between Train and Val!"
    assert val_df["TransactionDT"].max() <= test_df["TransactionDT"].min(), "Temporal leakage between Val and Test!"

    return train_df, val_df, test_df


def preprocess_tabular_features(
    train_df: pd.DataFrame, val_df: pd.DataFrame, test_df: pd.DataFrame
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[str]]:
    """Impute, encode, and scale features strictly fitted on the train split only."""
    metadata_cols = {"TransactionID", "isFraud", "TransactionDT", "card_id", "device_id"}
    candidate_cols = [c for c in train_df.columns if c not in metadata_cols]

    # Additional high-signal engineered features
    for split in [train_df, val_df, test_df]:
        split["log_TransactionAmt"] = np.log1p(np.maximum(0, split["TransactionAmt"]))
        split["TransactionAmt_decimal"] = split["TransactionAmt"] - np.floor(split["TransactionAmt"])

    feature_cols = [c for c in candidate_cols if c in train_df.columns] + ["log_TransactionAmt", "TransactionAmt_decimal"]

    # Identify categorical vs numerical columns
    categorical_cols = []
    numerical_cols = []
    for col in feature_cols:
        if train_df[col].dtype == "object" or train_df[col].dtype.name == "category":
            categorical_cols.append(col)
        else:
            numerical_cols.append(col)

    print(f"[*] Processing {len(numerical_cols)} numerical and {len(categorical_cols)} categorical features...")

    # 1. Fit numerical imputers (median) and Scaler strictly on TRAIN
    num_medians: Dict[str, float] = {}
    for col in numerical_cols:
        med = float(train_df[col].median())
        num_medians[col] = 0.0 if np.isnan(med) else med

    for split in [train_df, val_df, test_df]:
        for col in numerical_cols:
            split[col] = split[col].fillna(num_medians[col])

    scaler = StandardScaler()
    train_num_scaled = scaler.fit_transform(train_df[numerical_cols].values)
    val_num_scaled = scaler.transform(val_df[numerical_cols].values)
    test_num_scaled = scaler.transform(test_df[numerical_cols].values)

    # 2. Fit categorical frequency/label encoders strictly on TRAIN
    train_cat_encoded = []
    val_cat_encoded = []
    test_cat_encoded = []

    for col in categorical_cols:
        # Frequency encoding based strictly on train split
        freq_map = train_df[col].astype(str).value_counts(normalize=True).to_dict()
        train_cat_encoded.append(train_df[col].astype(str).map(freq_map).fillna(0.0).values[:, None])
        val_cat_encoded.append(val_df[col].astype(str).map(freq_map).fillna(0.0).values[:, None])
        test_cat_encoded.append(test_df[col].astype(str).map(freq_map).fillna(0.0).values[:, None])

    if categorical_cols:
        train_cat_arr = np.hstack(train_cat_encoded)
        val_cat_arr = np.hstack(val_cat_encoded)
        test_cat_arr = np.hstack(test_cat_encoded)

        X_train = np.hstack([train_num_scaled, train_cat_arr]).astype(np.float32)
        X_val = np.hstack([val_num_scaled, val_cat_arr]).astype(np.float32)
        X_test = np.hstack([test_num_scaled, test_cat_arr]).astype(np.float32)
    else:
        X_train = train_num_scaled.astype(np.float32)
        X_val = val_num_scaled.astype(np.float32)
        X_test = test_num_scaled.astype(np.float32)

    # Clean any leftover NaNs or Infs
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num(X_val, nan=0.0, posinf=0.0, neginf=0.0)
    X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

    y_train = train_df["isFraud"].values.astype(np.float32)
    y_val = val_df["isFraud"].values.astype(np.float32)
    y_test = test_df["isFraud"].values.astype(np.float32)

    all_feature_names = numerical_cols + [f"{c}_freq" for c in categorical_cols]
    return X_train, y_train, X_val, y_val, X_test, y_test, all_feature_names


def prepare_data(config: PipelineConfig) -> Tuple[Dict[str, pd.DataFrame], Dict[str, np.ndarray], List[str]]:
    """End-to-end data pipeline orchestrator."""
    config.ensure_dirs()
    set_seeds(config.seed)

    raw_df = load_raw_ieee_data(config.data_dir, sample_size=config.sample_size)
    train_df, val_df, test_df = perform_temporal_split(
        raw_df, train_ratio=config.train_ratio, val_ratio=config.val_ratio
    )

    X_train, y_train, X_val, y_val, X_test, y_test, feature_names = preprocess_tabular_features(
        train_df, val_df, test_df
    )

    # Save processed splits and metadata
    arrays = {
        "X_train": X_train, "y_train": y_train,
        "X_val": X_val, "y_val": y_val,
        "X_test": X_test, "y_test": y_test,
    }
    np.savez_compressed(config.out_dir / "processed_arrays.npz", **arrays)

    # Save DataFrame splits for graph construction
    dfs = {"train": train_df, "val": val_df, "test": test_df}
    for name, df in dfs.items():
        df[["TransactionID", "TransactionDT", "isFraud", "card_id", "device_id"]].to_parquet(
            config.out_dir / f"{name}_meta.parquet", index=False
        )

    with open(config.out_dir / "feature_names.json", "w") as f:
        json.dump(feature_names, f, indent=2)

    print(f"[+] Successfully saved preprocessed data to {config.out_dir}")
    return dfs, arrays, feature_names


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare IEEE-CIS data with strict temporal splitting.")
    parser.add_argument("--data_dir", type=Path, default=get_default_data_dir(), help="Path to raw IEEE-CIS dataset")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output path for artifacts")
    parser.add_argument("--sample", type=int, default=None, help="Optional sample size for smoke testing")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(data_dir=args.data_dir, out_dir=args.out_dir, sample_size=args.sample, seed=args.seed)
    prepare_data(cfg)
