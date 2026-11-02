"""End-to-End Orchestrator for Uncertainty-Aware Graph Fraud Learning Pipeline.

Executes all pipeline stages sequentially:
1. Data Preparation & Temporal Splitting (dl.src.data)
2. Graph Construction & Hub Degree Capping (dl.src.graph)
3. Tabular Baselines: LightGBM, XGBoost, MLP, LightGBM+Graph (dl.src.baselines)
4. Unsupervised Autoencoder Anomaly Scoring (dl.src.autoencoder)
5. GraphSAGE Neural Training & Ablations (dl.src.gnn)
6. Monte Carlo Dropout Uncertainty Quantification (dl.src.mc_dropout)
7. Post-Hoc Probability Calibration (dl.src.calibration)
8. Uncertainty-Driven Next-Best-Action Policy (dl.src.decision)
9. Comprehensive Metrics, Calibration Diagrams, and Plots (dl.src.evaluate)
10. Multi-Seed Aggregation (Mean +/- Std).

Task 8: Added --start_stage / --end_stage to resume from saved artifacts.
Task 9: Per-stage timings and peak CPU / GPU memory printed after each stage.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Ensure repo root is on sys.path for direct script execution
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import numpy as np
import pandas as pd
import psutil
import torch

from dl.src.autoencoder import train_autoencoder
from dl.src.baselines import run_baselines
from dl.src.calibration import calibrate_predictions
from dl.src.config import (
    PipelineConfig,
    get_default_checkpoint_dir,
    get_default_data_dir,
    get_default_output_dir,
    save_json,
    set_seeds,
)
from dl.src.data import prepare_data
from dl.src.decision import evaluate_decision_policy
from dl.src.evaluate import generate_evaluation_artifacts
from dl.src.gnn import run_gnn_pipeline
from dl.src.graph import build_graph
from dl.src.mc_dropout import compute_mc_uncertainties

# ---------------------------------------------------------------------------
# Stage registry — used for --start_stage / --end_stage
# ---------------------------------------------------------------------------

STAGES = [
    ("data",        "Data Preparation & Temporal Splitting"),
    ("graph",       "Graph Construction & Hub Degree Capping"),
    ("baselines",   "Tabular Baselines (LightGBM, XGBoost, Tabular MLP)"),
    ("autoencoder", "Autoencoder Anomaly Representation Learning"),
    ("gnn",         "Graph Neural Network (GraphSAGE) Training & Ablations"),
    ("mc_dropout",  "Monte Carlo Dropout Inference"),
    ("calibration", "Post-Hoc Probability Calibration (ECE Reduction)"),
    ("decision",    "Uncertainty-Driven Next-Best-Action Policy Evaluation"),
    ("evaluate",    "Comprehensive Evaluation, Calibration Diagrams, and Plots"),
]

STAGE_NAMES = [s[0] for s in STAGES]

# Artifacts that must exist for each stage to resume from it.
STAGE_RESUME_ARTIFACTS = {
    "graph":       ["processed_arrays.npz", "train_meta.parquet", "val_meta.parquet", "test_meta.parquet"],
    "baselines":   ["edge_index.pt", "graph_features.npz"],
    "autoencoder": ["preds_test_lgb.npy", "preds_val_lgb.npy"],
    "gnn":         ["autoencoder_features.npz"],
    "mc_dropout":  ["preds_test_gnn_ae.npy", "preds_val_gnn_ae.npy"],
    "calibration": ["mc_dropout_val.npz", "mc_dropout_test.npz"],
    "decision":    ["mc_dropout_val.npz", "mc_dropout_test.npz"],
    "evaluate":    ["decision_policy_summary.json"],
}


def check_resume_artifacts(stage_name: str, config: PipelineConfig) -> None:
    """Raise a clear error if artifacts required to resume from stage_name are missing."""
    required = STAGE_RESUME_ARTIFACTS.get(stage_name, [])
    missing = [f for f in required if not (config.out_dir / f).exists()]
    if missing:
        raise FileNotFoundError(
            f"\n[ERROR] Cannot resume from stage '{stage_name}': missing required artifacts in {config.out_dir}:\n"
            + "\n".join(f"  - {f}" for f in missing)
            + "\nRe-run from an earlier stage or from the beginning (omit --start_stage)."
        )


def _memory_report() -> str:
    """Return a string with current CPU RSS and GPU peak memory."""
    proc = psutil.Process(os.getpid())
    cpu_rss_mb = proc.memory_info().rss / 1024 ** 2
    msg = f"CPU RSS: {cpu_rss_mb:.0f} MB"
    if torch.cuda.is_available():
        gpu_peak_mb = torch.cuda.max_memory_allocated() / 1024 ** 2
        torch.cuda.reset_peak_memory_stats()
        msg += f"  |  GPU Peak: {gpu_peak_mb:.0f} MB"
    return msg


def run_stage(name: str, fn, config: PipelineConfig, timings: dict):
    """Run a single pipeline stage, measure elapsed time, and print memory stats."""
    t0 = time.time()
    result = fn(config)
    elapsed = time.time() - t0
    timings[name] = elapsed
    print(f"\n  [STAGE TIMING] '{name}' completed in {elapsed:.1f}s ({elapsed/60:.2f} min)")
    print(f"  [MEMORY] {_memory_report()}")
    return result


def run_single_seed_pipeline(
    config: PipelineConfig,
    start_stage: str = "data",
    end_stage: str = "evaluate",
    prob_source: str = "ensemble",
) -> pd.DataFrame:
    """Run the complete pipeline for a single random seed, optionally resuming mid-pipeline."""
    print("\n" + "=" * 90)
    print(f"       EXECUTING PIPELINE (Seed: {config.seed} | Device: {config.device.upper()})")
    print(f"       Stages: [{start_stage}] -> [{end_stage}]")
    print("=" * 90)

    overall_start = time.time()
    set_seeds(config.seed)
    config.ensure_dirs()

    start_idx = STAGE_NAMES.index(start_stage)
    end_idx = STAGE_NAMES.index(end_stage)

    if start_idx > 0:
        print(f"\n[*] Resuming from stage '{start_stage}' — checking required artifacts...")
        check_resume_artifacts(start_stage, config)
        print(f"[+] All required artifacts present. Skipping stages before '{start_stage}'.")

    active_stages = STAGE_NAMES[start_idx: end_idx + 1]
    n_total = len(active_stages)
    timings = {}
    df_metrics = pd.DataFrame()

    stage_fns = {
        "data":        lambda cfg: prepare_data(cfg),
        "graph":       lambda cfg: build_graph(cfg),
        "baselines":   lambda cfg: run_baselines(cfg),
        "autoencoder": lambda cfg: train_autoencoder(cfg),
        "gnn":         lambda cfg: run_gnn_pipeline(cfg),
        "mc_dropout":  lambda cfg: compute_mc_uncertainties(cfg),
        "calibration": lambda cfg: calibrate_predictions(cfg),
        "decision":    lambda cfg: evaluate_decision_policy(cfg, prob_source=prob_source),
        "evaluate":    lambda cfg: generate_evaluation_artifacts(cfg),
    }

    for i, stage_name in enumerate(active_stages, start=1):
        stage_label = dict(STAGES)[stage_name]
        print(f"\n[STAGE {i}/{n_total}] {stage_label}...")
        result = run_stage(stage_name, stage_fns[stage_name], config, timings)
        if stage_name == "evaluate":
            df_metrics = result

    # Print stage timing table
    print("\n" + "=" * 60)
    print("              PER-STAGE TIMING SUMMARY")
    print("=" * 60)
    for stage_name, elapsed in timings.items():
        label = dict(STAGES).get(stage_name, stage_name)
        print(f"  {label:45s}: {elapsed:6.1f}s  ({elapsed/60:.2f} min)")
    total_elapsed = time.time() - overall_start
    print(f"  {'TOTAL':45s}: {total_elapsed:6.1f}s  ({total_elapsed/60:.2f} min)")
    print("=" * 60)

    save_json(config.out_dir / "stage_timings.json", timings)

    # Leakage guardrail
    if not df_metrics.empty and "ROC-AUC" in df_metrics.columns:
        max_roc = df_metrics["ROC-AUC"].max()
        if max_roc > 0.98:
            print(f"\n[!] SANITY CHECK ALERT: Max Test ROC-AUC is {max_roc:.4f} (> 0.98). Check for leakage.")
        else:
            print(f"\n[+] Leakage Sanity Check Passed (Max Test ROC-AUC: {max_roc:.4f} <= 0.98).")

    print(f"\n[checkmark] Seed {config.seed} completed in {total_elapsed / 60:.2f} minutes.")
    return df_metrics


def run_multi_seed_pipeline(
    config: PipelineConfig,
    seeds: List[int],
    start_stage: str = "data",
    end_stage: str = "evaluate",
    prob_source: str = "ensemble",
) -> None:
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
        df = run_single_seed_pipeline(seed_cfg, start_stage=start_stage, end_stage=end_stage, prob_source=prob_source)
        df["Seed"] = seed
        seed_dfs.append(df)

    if not seed_dfs or all(df.empty for df in seed_dfs):
        print("[!] No metrics to aggregate (evaluate stage may not have been run).")
        return

    # Aggregate across seeds
    all_runs = pd.concat(seed_dfs, ignore_index=True)
    numeric_cols = [c for c in all_runs.columns if c not in ["Model", "Seed", "Calibrated"]]

    grouped = all_runs.groupby("Model")[numeric_cols]
    means = grouped.mean()
    stds = grouped.std().fillna(0.0)

    summary_rows = []
    for model_name in means.index:
        row = {"Model": model_name}
        for col in numeric_cols:
            m = means.loc[model_name, col]
            s = stds.loc[model_name, col]
            row[col] = f"{m:.4f} +/- {s:.4f}"
        summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(base_out_dir / "multi_seed_summary.csv", index=False)

    # Markdown table
    md_lines = ["# Multi-Seed Aggregate Results\n"]
    md_lines.append(f"Seeds: {seeds}\n")
    md_lines.append(df_summary.to_markdown(index=False))
    md_lines.append("\n\n> Values are **Mean +/- Std** across seeds.\n")
    (base_out_dir / "multi_seed_summary.md").write_text("\n".join(md_lines), encoding="utf-8")

    print("\n" + "=" * 90)
    print(f"             MULTI-SEED STATISTICAL SUMMARY ({len(seeds)} Seeds: {seeds})")
    print("=" * 90)
    print(df_summary.to_string(index=False))
    print("=" * 90)
    print(f"[+] Saved multi-seed summary to {base_out_dir / 'multi_seed_summary.csv'}")
    print(f"[+] Saved markdown table to {base_out_dir / 'multi_seed_summary.md'}")


def main():
    parser = argparse.ArgumentParser(
        description="Run complete Deep Learning Fraud Investigation Pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--data_dir", type=Path, default=get_default_data_dir(),
                        help="Path to raw IEEE-CIS dataset (required for stage 'data')")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(),
                        help="Artifact output directory")
    parser.add_argument("--ckpt_dir", type=Path, default=get_default_checkpoint_dir(),
                        help="Checkpoint directory")
    parser.add_argument("--sample", type=int, default=None,
                        help="Sample size for smoke testing (e.g. 100000)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42],
                        help="Random seeds (e.g. --seeds 42 123 456)")
    parser.add_argument("--mc_samples", type=int, default=30,
                        help="Number of MC Dropout passes")
    parser.add_argument(
        "--start_stage", type=str, default="data", choices=STAGE_NAMES,
        help=(
            "Stage to start from (default: 'data'). Use to resume from saved artifacts. "
            f"Choices: {STAGE_NAMES}"
        ),
    )
    parser.add_argument(
        "--end_stage", type=str, default="evaluate", choices=STAGE_NAMES,
        help=(
            "Stage to stop after (default: 'evaluate'). "
            f"Choices: {STAGE_NAMES}"
        ),
    )
    parser.add_argument(
        "--prob_source", type=str, default="ensemble",
        choices=["gnn", "lgb_graph", "ensemble"],
        help="Probability source for the decision policy (default: 'ensemble').",
    )
    args = parser.parse_args()

    # Validate stage ordering
    start_idx = STAGE_NAMES.index(args.start_stage)
    end_idx = STAGE_NAMES.index(args.end_stage)
    if start_idx > end_idx:
        parser.error(f"--start_stage '{args.start_stage}' must come before --end_stage '{args.end_stage}'.")

    cfg = PipelineConfig(
        data_dir=args.data_dir,
        out_dir=args.out_dir,
        checkpoint_dir=args.ckpt_dir,
        sample_size=args.sample,
        seeds=args.seeds,
        mc_samples=args.mc_samples,
    )

    if len(args.seeds) > 1:
        run_multi_seed_pipeline(
            cfg, args.seeds,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
            prob_source=args.prob_source,
        )
    else:
        cfg.seed = args.seeds[0]
        run_single_seed_pipeline(
            cfg,
            start_stage=args.start_stage,
            end_stage=args.end_stage,
            prob_source=args.prob_source,
        )


if __name__ == "__main__":
    main()
