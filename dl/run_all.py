"""End-to-End Orchestrator for Uncertainty-Aware Graph Fraud Learning Pipeline.

Executes all pipeline stages sequentially:
1. Data Preparation & Temporal Splitting (dl.src.data)
2. Graph Construction & Hub Degree Capping (dl.src.graph)
3. Tabular Baselines: LightGBM, XGBoost, MLP, LightGBM+Graph (dl.src.baselines)
4. Unsupervised Autoencoder Anomaly Scoring (dl.src.autoencoder)
5. GraphSAGE Neural Training & Ablations (dl.src.gnn)
6. Monte Carlo Dropout Uncertainty Quantification (dl.src.mc_dropout)
7. Uncertainty-Driven Next-Best-Action Policy (dl.src.decision)
8. Comprehensive Metrics, Calibration Diagrams, and Plots (dl.src.evaluate)
9. Multi-Seed Aggregation (Mean +/- Std).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

from dl.src.autoencoder import train_autoencoder
from dl.src.baselines import run_baselines
from dl.src.config import (
    PipelineConfig,
    get_default_checkpoint_dir,
    get_default_data_dir,
    get_default_output_dir,
    set_seeds,
)
from dl.src.data import prepare_data
from dl.src.decision import evaluate_decision_policy
from dl.src.evaluate import generate_evaluation_artifacts
from dl.src.gnn import run_gnn_pipeline
from dl.src.graph import build_graph
from dl.src.mc_dropout import compute_mc_uncertainties


def run_single_seed_pipeline(config: PipelineConfig) -> pd.DataFrame:
    """Run the complete pipeline for a single random seed."""
    print("\n" + "=" * 90)
    print(f"       EXECUTING PIPELINE (Seed: {config.seed} | Device: {config.device.upper()})")
    print("=" * 90)

    start_time = time.time()
    set_seeds(config.seed)
    config.ensure_dirs()

    # Stage 1: Data Preparation & Temporal Split
    print("\n[STAGE 1/8] Data Loading & Temporal Splitting...")
    prepare_data(config)

    # Stage 2: Graph Construction
    print("\n[STAGE 2/8] Graph Construction & Hub Degree Capping...")
    build_graph(config)

    # Stage 3: Tabular Baselines
    print("\n[STAGE 3/8] Tabular Baselines (LightGBM, XGBoost, Tabular MLP)...")
    run_baselines(config)

    # Stage 4: Autoencoder
    print("\n[STAGE 4/8] Autoencoder Anomaly Representation Learning...")
    train_autoencoder(config)

    # Stage 5: GNN (GraphSAGE) Training
    print("\n[STAGE 5/8] Graph Neural Network (GraphSAGE) Training & Ablations...")
    run_gnn_pipeline(config)

    # Stage 6: Monte Carlo Dropout
    print("\n[STAGE 6/8] Monte Carlo Dropout Inference (T=30 Stochastic Passes)...")
    compute_mc_uncertainties(config)

    # Stage 7: Next-Best-Action Decision Policy
    print("\n[STAGE 7/8] Uncertainty-Driven Next-Best-Action Policy Evaluation...")
    evaluate_decision_policy(config)

    # Stage 8: Evaluation & Plots
    print("\n[STAGE 8/8] Evaluation, Calibration, and Visualization Artifacts...")
    df_metrics = generate_evaluation_artifacts(config)

    # Sanity Check: Test ROC-AUC Leakage Guardrail
    max_roc = df_metrics["ROC-AUC"].max()
    if max_roc > 0.98:
        print(f"\n[!] SANITY CHECK ALERT: Max Test ROC-AUC is {max_roc:.4f} (> 0.98).")
        print("    Investigate possible target or temporal leakage before proceeding.")
    else:
        print(f"\n[+] Leakage Sanity Check Passed (Max Test ROC-AUC: {max_roc:.4f} <= 0.98).")

    elapsed = time.time() - start_time
    print(f"\n[✓] Seed {config.seed} completed in {elapsed / 60:.2f} minutes.")
    return df_metrics


def run_multi_seed_pipeline(config: PipelineConfig, seeds: List[int]) -> None:
    """Run pipeline across multiple seeds and generate statistical Mean +/- Std report."""
    seed_dfs = []
    base_out_dir = config.out_dir

    for seed in seeds:
        seed_cfg = PipelineConfig(
            data_dir=config.data_dir,
            out_dir=base_out_dir / f"seed_{seed}",
            checkpoint_dir=config.checkpoint_dir / f"seed_{seed}",
            device=config.device,
            seed=seed,
            sample_size=config.sample_size,
            mc_samples=config.mc_samples,
        )
        df = run_single_seed_pipeline(seed_cfg)
        df["Seed"] = seed
        seed_dfs.append(df)

    # Aggregate across seeds
    all_runs = pd.concat(seed_dfs, ignore_index=True)
    numeric_cols = [c for c in all_runs.columns if c not in ["Model", "Seed"]]

    grouped = all_runs.groupby("Model")[numeric_cols]
    means = grouped.mean()
    stds = grouped.std().fillna(0.0)

    summary_rows = []
    for model_name in means.index:
        row = {"Model": model_name}
        for col in numeric_cols:
            m = means.loc[model_name, col]
            s = stds.loc[model_name, col]
            row[col] = f"{m:.4f} ± {s:.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(base_out_dir / "multi_seed_summary.csv", index=False)

    print("\n" + "=" * 90)
    print(f"             MULTI-SEED STATISTICAL SUMMARY ({len(seeds)} Seeds: {seeds})")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)
    print(f"[+] Multi-seed summary saved to {base_out_dir / 'multi_seed_summary.csv'}")


def main():
    parser = argparse.ArgumentParser(description="Run complete Deep Learning Fraud Investigation Pipeline.")
    parser.add_argument("--data_dir", type=Path, default=get_default_data_dir(), help="Path to raw IEEE-CIS dataset")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Artifact output directory")
    parser.add_argument("--ckpt_dir", type=Path, default=get_default_checkpoint_dir(), help="Checkpoint directory")
    parser.add_argument("--sample", type=int, default=None, help="Sample size for smoke testing (e.g. 20000)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42], help="Random seeds (e.g. --seeds 42 123 456)")
    parser.add_argument("--mc_samples", type=int, default=30, help="Number of MC Dropout passes")
    args = parser.parse_args()

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        checkpoint_dir=args.ckpt_dir,
        sample_size=args.sample,
        seeds=args.seeds,
        mc_samples=args.mc_samples,
    )

    if len(args.seeds) > 1:
        run_multi_seed_pipeline(cfg, args.seeds)
    else:
        cfg.seed = args.seeds[0]
        run_single_seed_pipeline(cfg)


if __name__ == "__main__":
    main()
