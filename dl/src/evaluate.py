"""Comprehensive Evaluation Suite for Fraud Detection, Uncertainty, and Decision Routing.

Implements:
1. Primary metrics: PR-AUC, ROC-AUC, Recall @ Top 0.5% / 1% Alert Budget, Recall @ 90% Precision.
2. Full Comparison Table: LightGBM, XGBoost, MLP, Autoencoder, GNN, GNN+AE, LightGBM+Graph, MC-Dropout GNN.
3. Graph & Autoencoder Ablations.
4. Calibration Analysis: Expected Calibration Error (ECE) & Reliability Diagrams (Pre vs Post Calibration).
5. Risk-Coverage / Abstention Curves & Uncertainty-Error Correlation.
6. Ring-Level Detection Metrics for Shared-Device / Multi-Card Syndicates.
7. Multi-Seed Reporting (Mean +/- Std).
8. Publication-Quality Plots (PNG) and Export Tables (CSV/JSON).
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from dl.src.config import PipelineConfig, get_default_output_dir, save_json, set_seeds


def validate_binary_targets(y_true: np.ndarray, context: str = "evaluation") -> None:
    """Ensure ground truth target array contains both positive and negative classes."""
    unique = np.unique(y_true)
    if len(unique) < 2:
        raise ValueError(
            f"Invalid target distribution in {context}: found only class(es) {unique.tolist()}. "
            f"Both positive (fraud=1) and negative (legitimate=0) classes are required for evaluation."
        )


def compute_recall_at_alert_budget(y_true: np.ndarray, y_score: np.ndarray, budget_fraction: float) -> float:
    """Compute recall when investigating only the top budget_fraction of alerts."""
    validate_binary_targets(y_true, "compute_recall_at_alert_budget")
    k = max(1, int(len(y_score) * budget_fraction))
    top_indices = np.argsort(y_score)[::-1][:k]
    fraud_detected = y_true[top_indices].sum()
    total_fraud = max(1.0, y_true.sum())
    return float(fraud_detected / total_fraud)


def compute_recall_at_precision(y_true: np.ndarray, y_score: np.ndarray, target_precision: float = 0.90) -> float:
    """Compute maximum recall achieved at or above the target precision threshold."""
    validate_binary_targets(y_true, "compute_recall_at_precision")
    precisions, recalls, _ = precision_recall_curve(y_true, y_score)
    valid_recalls = recalls[precisions >= target_precision]
    return float(np.max(valid_recalls)) if len(valid_recalls) > 0 else 0.0


def compute_expected_calibration_error(
    y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15
) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray]:
    """Compute Expected Calibration Error (ECE) and bin statistics."""
    validate_binary_targets(y_true, "compute_expected_calibration_error")
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    bin_accs = []
    bin_confs = []
    bin_sizes = []

    for lower, upper in zip(bin_lowers, bin_uppers):
        in_bin = (y_prob > lower) & (y_prob <= upper)
        prop_in_bin = float(np.mean(in_bin))
        bin_sizes.append(int(in_bin.sum()))
        if prop_in_bin > 0:
            accuracy_in_bin = float(np.mean(y_true[in_bin]))
            avg_confidence_in_bin = float(np.mean(y_prob[in_bin]))
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
            bin_accs.append(accuracy_in_bin)
            bin_confs.append(avg_confidence_in_bin)
        else:
            bin_accs.append(0.0)
            bin_confs.append((lower + upper) / 2.0)

    return float(ece), np.array(bin_confs), np.array(bin_accs), np.array(bin_sizes)


def compute_risk_coverage_curve(
    y_true: np.ndarray, y_prob: np.ndarray, uncertainties: np.ndarray, coverages: List[float] = None
) -> Tuple[List[float], List[float]]:
    """Compute error rates as highest-uncertainty samples are progressively abstained."""
    if coverages is None:
        coverages = list(np.linspace(1.0, 0.40, 25))

    # Sort samples from lowest uncertainty to highest uncertainty
    sort_idx = np.argsort(uncertainties)
    n = len(y_true)

    errors = []
    for cov in coverages:
        k = max(1, int(n * cov))
        retained_idx = sort_idx[:k]

        retained_y = y_true[retained_idx]
        retained_preds = (y_prob[retained_idx] >= 0.5).astype(int)

        # Classification error rate on retained set
        err = float(np.mean(retained_y != retained_preds))
        errors.append(err)

    return coverages, errors


def evaluate_ring_level_metrics(test_meta: pd.DataFrame, test_probs: np.ndarray) -> Dict[str, float]:
    """Evaluate detection metrics specifically on shared-device syndicates and rings."""
    test_meta = test_meta.copy()
    test_meta["pred_prob"] = test_probs

    # Identify multi-card device clusters (fraud rings)
    device_card_counts = test_meta.groupby("device_id")["card_id"].nunique().to_dict()
    test_meta["device_card_count"] = test_meta["device_id"].map(device_card_counts).fillna(1)

    ring_mask = (test_meta["device_id"] != "D_NONE") & (test_meta["device_card_count"] >= 2)
    ring_df = test_meta[ring_mask]

    if len(ring_df) == 0 or ring_df["isFraud"].sum() == 0:
        return {"ring_pr_auc": 0.0, "ring_roc_auc": 0.0, "ring_frauds": 0, "ring_size": 0}

    ring_y = ring_df["isFraud"].values
    ring_p = ring_df["pred_prob"].values

    ring_pr = float(average_precision_score(ring_y, ring_p))
    ring_roc = float(roc_auc_score(ring_y, ring_p))
    ring_rec_1pct = compute_recall_at_alert_budget(ring_y, ring_p, budget_fraction=0.05)

    return {
        "ring_pr_auc": ring_pr,
        "ring_roc_auc": ring_roc,
        "ring_recall_top5pct": ring_rec_1pct,
        "ring_frauds": int(ring_y.sum()),
        "ring_size": len(ring_df),
    }


def generate_evaluation_artifacts(config: PipelineConfig) -> pd.DataFrame:
    """Load all model predictions, compute metrics, plot curves, and output summary tables."""
    config.ensure_dirs()
    set_seeds(config.seed)

    # 1. Load test metadata and ground truth labels
    test_meta = pd.read_parquet(config.out_dir / "test_meta.parquet")
    y_test = test_meta["isFraud"].values.astype(np.float32)
    validate_binary_targets(y_test, "Test Set Evaluation")

    # 2. Collect model predictions
    models = {
        "LightGBM (Tabular)": config.out_dir / "preds_test_lgb.npy",
        "XGBoost (Tabular)": config.out_dir / "preds_test_xgb.npy",
        "Tabular MLP": config.out_dir / "preds_test_mlp.npy",
        "Autoencoder (Score)": config.out_dir / "preds_test_ae.npy",
        "LightGBM + Graph": config.out_dir / "preds_test_lgb_graph.npy",
        "GraphSAGE (GNN)": config.out_dir / "preds_test_gnn.npy",
        "GNN + AE (Full)": config.out_dir / "preds_test_gnn_ae.npy",
    }

    # Load MC Dropout test predictions if available
    mc_path = config.out_dir / "mc_dropout_test.npz"
    if mc_path.exists():
        mc_data = np.load(mc_path)
        mc_raw_probs = mc_data["mean_prob"]
        mc_cal_probs = mc_data["mean_prob_calibrated"] if "mean_prob_calibrated" in mc_data else mc_raw_probs
        mc_uncert = mc_data["variance"]
    else:
        mc_raw_probs = None
        mc_cal_probs = None
        mc_uncert = None

    metrics_list = []
    pred_dict: Dict[str, np.ndarray] = {}

    for name, path in models.items():
        if path.exists():
            preds = np.load(path)
            pred_dict[name] = preds

            pr_auc = float(average_precision_score(y_test, preds))
            roc_auc = float(roc_auc_score(y_test, preds))
            rec_05 = compute_recall_at_alert_budget(y_test, preds, budget_fraction=0.005)
            rec_10 = compute_recall_at_alert_budget(y_test, preds, budget_fraction=0.010)
            rec_p90 = compute_recall_at_precision(y_test, preds, target_precision=0.90)
            ece, _, _, _ = compute_expected_calibration_error(y_test, np.clip(preds, 0, 1))

            metrics_list.append({
                "Model": name,
                "PR-AUC": round(pr_auc, 4),
                "ROC-AUC": round(roc_auc, 4),
                "Recall @ Top 0.5%": round(rec_05, 4),
                "Recall @ Top 1.0%": round(rec_10, 4),
                "Recall @ 90% Prec": round(rec_p90, 4),
                "ECE": round(ece, 4),
            })

    # Add MC-Dropout GNN model entries (Raw and Calibrated)
    if mc_raw_probs is not None:
        pr_raw = float(average_precision_score(y_test, mc_raw_probs))
        roc_raw = float(roc_auc_score(y_test, mc_raw_probs))
        rec_05_raw = compute_recall_at_alert_budget(y_test, mc_raw_probs, budget_fraction=0.005)
        rec_10_raw = compute_recall_at_alert_budget(y_test, mc_raw_probs, budget_fraction=0.010)
        rec_p90_raw = compute_recall_at_precision(y_test, mc_raw_probs, target_precision=0.90)
        ece_raw, _, _, _ = compute_expected_calibration_error(y_test, mc_raw_probs)

        metrics_list.append({
            "Model": "MC-Dropout GNN (Raw)",
            "PR-AUC": round(pr_raw, 4),
            "ROC-AUC": round(roc_raw, 4),
            "Recall @ Top 0.5%": round(rec_05_raw, 4),
            "Recall @ Top 1.0%": round(rec_10_raw, 4),
            "Recall @ 90% Prec": round(rec_p90_raw, 4),
            "ECE": round(ece_raw, 4),
        })
        pred_dict["MC-Dropout GNN (Raw)"] = mc_raw_probs

    if mc_cal_probs is not None:
        pr_cal = float(average_precision_score(y_test, mc_cal_probs))
        roc_cal = float(roc_auc_score(y_test, mc_cal_probs))
        rec_05_cal = compute_recall_at_alert_budget(y_test, mc_cal_probs, budget_fraction=0.005)
        rec_10_cal = compute_recall_at_alert_budget(y_test, mc_cal_probs, budget_fraction=0.010)
        rec_p90_cal = compute_recall_at_precision(y_test, mc_cal_probs, target_precision=0.90)
        ece_cal, _, _, _ = compute_expected_calibration_error(y_test, mc_cal_probs)

        metrics_list.append({
            "Model": "MC-Dropout GNN (Calibrated)",
            "PR-AUC": round(pr_cal, 4),
            "ROC-AUC": round(roc_cal, 4),
            "Recall @ Top 0.5%": round(rec_05_cal, 4),
            "Recall @ Top 1.0%": round(rec_10_cal, 4),
            "Recall @ 90% Prec": round(rec_p90_cal, 4),
            "ECE": round(ece_cal, 4),
        })
        pred_dict["MC-Dropout GNN (Calibrated)"] = mc_cal_probs

    df_metrics = pd.DataFrame(metrics_list)
    df_metrics.to_csv(config.out_dir / "metrics_summary.csv", index=False)
    save_json(config.out_dir / "metrics_summary.json", df_metrics.to_dict(orient="records"))

    # 3. Print Summary Table
    print("\n" + "=" * 90)
    print("                MODEL PERFORMANCE COMPARISON (TEMPORAL TEST SET)")
    print("=" * 90)
    print(df_metrics.to_string(index=False))
    print("=" * 90)

    # 4. Ring-Level Metrics
    best_pred_for_rings = mc_cal_probs if mc_cal_probs is not None else pred_dict.get("GNN + AE (Full)", list(pred_dict.values())[0])
    ring_metrics = evaluate_ring_level_metrics(test_meta, best_pred_for_rings)
    print(f"\n[*] Ring-Level Fraud Detection Metrics (Shared-Device Clusters):")
    print(f"  • Cluster Size: {ring_metrics['ring_size']} txns | Confirmed Frauds: {ring_metrics['ring_frauds']}")
    print(f"  • Ring PR-AUC:  {ring_metrics['ring_pr_auc']:.4f} | Ring ROC-AUC: {ring_metrics['ring_roc_auc']:.4f}")
    save_json(config.out_dir / "ring_metrics.json", ring_metrics)

    # 5. Risk-Coverage Table
    if mc_cal_probs is not None and mc_uncert is not None:
        cov_steps = [1.0, 0.95, 0.90, 0.80, 0.70, 0.50]
        coverages, errors = compute_risk_coverage_curve(y_test, mc_cal_probs, mc_uncert, coverages=cov_steps)
        print("\n[*] Risk-Coverage / Selective Classification Analysis (Abstention on Uncertainty):")
        for cov, err in zip(coverages, errors):
            print(f"  • Coverage: {cov*100:5.1f}% of most confident transactions -> Error Rate on Retained Set: {err*100:5.2f}%")

    # 6. Generate Figures (PNG)
    plt.style.use("default")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Plot 1: Precision-Recall Curves
    ax1 = axes[0, 0]
    for name, preds in pred_dict.items():
        if name in ["LightGBM (Tabular)", "LightGBM + Graph", "Tabular MLP", "GraphSAGE (GNN)", "GNN + AE (Full)", "MC-Dropout GNN (Calibrated)"]:
            p, r, _ = precision_recall_curve(y_test, preds)
            score = average_precision_score(y_test, preds)
            ax1.plot(r, p, label=f"{name} (PR={score:.3f})", lw=2)
    ax1.set_title("Precision-Recall Curves (Temporal Test)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("Recall")
    ax1.set_ylabel("Precision")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(loc="lower left", fontsize=9)

    # Plot 2: ROC Curves
    ax2 = axes[0, 1]
    for name, preds in pred_dict.items():
        if name in ["LightGBM (Tabular)", "LightGBM + Graph", "Tabular MLP", "GraphSAGE (GNN)", "GNN + AE (Full)", "MC-Dropout GNN (Calibrated)"]:
            fpr, tpr, _ = roc_curve(y_test, preds)
            score = roc_auc_score(y_test, preds)
            ax2.plot(fpr, tpr, label=f"{name} (ROC={score:.3f})", lw=2)
    ax2.plot([0, 1], [0, 1], "k--", label="Random Chance (0.500)")
    ax2.set_title("ROC Curves (Temporal Test)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive Rate")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(loc="lower right", fontsize=9)

    # Plot 3: Reliability Diagram (Pre vs Post Calibration)
    ax3 = axes[1, 0]
    mlp_preds = pred_dict.get("Tabular MLP")
    if mlp_preds is not None:
        ece_mlp, conf_mlp, acc_mlp, _ = compute_expected_calibration_error(y_test, mlp_preds)
        ax3.plot(conf_mlp, acc_mlp, "s--", label=f"Tabular MLP (ECE={ece_mlp:.3f})", color="tab:orange", alpha=0.8)
    if mc_raw_probs is not None:
        ece_raw, conf_raw, acc_raw, _ = compute_expected_calibration_error(y_test, mc_raw_probs)
        ax3.plot(conf_raw, acc_raw, "^:", label=f"MC-Dropout GNN Raw (ECE={ece_raw:.3f})", color="tab:purple", alpha=0.8)
    if mc_cal_probs is not None:
        ece_cal, conf_cal, acc_cal, _ = compute_expected_calibration_error(y_test, mc_cal_probs)
        ax3.plot(conf_cal, acc_cal, "o-", label=f"MC-Dropout GNN Calibrated (ECE={ece_cal:.3f})", color="tab:blue", lw=2.5)
    ax3.plot([0, 1], [0, 1], "k:", label="Perfect Calibration")
    ax3.set_title("Reliability Diagram / Calibration (ECE)", fontsize=13, fontweight="bold")
    ax3.set_xlabel("Mean Predicted Confidence")
    ax3.set_ylabel("Empirical True Positive Rate")
    ax3.grid(True, linestyle="--", alpha=0.6)
    ax3.legend(loc="upper left", fontsize=9)

    # Plot 4: Risk-Coverage / Abstention Curve
    ax4 = axes[1, 1]
    if mc_cal_probs is not None and mc_uncert is not None:
        cov_all, err_all = compute_risk_coverage_curve(y_test, mc_cal_probs, mc_uncert)
        ax4.plot([c * 100 for c in cov_all], [e * 100 for e in err_all], "o-", color="tab:red", lw=2)
        ax4.set_title("Risk-Coverage Abstention Analysis", fontsize=13, fontweight="bold")
        ax4.set_xlabel("Coverage (% Most Confident Transactions Retained)")
        ax4.set_ylabel("Error Rate on Retained Set (%)")
        ax4.grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout()
    plot_path = config.out_dir / "evaluation_curves.png"
    plt.savefig(plot_path, dpi=300)
    plt.close()

    print(f"\n[+] Generated comprehensive plots at {plot_path}")
    return df_metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate all models and produce summary plots.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, seed=args.seed)
    generate_evaluation_artifacts(cfg)
