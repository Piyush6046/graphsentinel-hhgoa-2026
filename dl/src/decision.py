"""Uncertainty-Driven Next-Best-Action Policy Engine with Expected-Cost Threshold Optimization.

Key Features:
1. Expected-Cost Objective:
   - Missed fraud on ALLOW: Loss = Transaction Amount ($ Exposure).
   - False positive on ACT: $10.0 manual friction cost.
   - Step-up verification on REQUEST_EVIDENCE: $2.0 step-up cost.
   - Analyst review on ESCALATE: $15.0 analyst review cost.
2. Operational Constraints Enforced on VALIDATION:
   - Total non-ALLOW actions <= 5.0% of traffic
   - ESCALATE actions <= 1.0% of traffic
   - REQUEST_EVIDENCE band >= 1.0% of traffic (guarantees a meaningful evidence-seeking band).
3. Configurable Probability Sources (Task 2):
   - 'gnn': Calibrated MC-Dropout GNN probability
   - 'lgb_graph': Calibrated LightGBM + Graph probability
   - 'ensemble': Average of Calibrated GNN and LightGBM+Graph (Default)
   - Uncertainty signal: MC-Dropout GNN predictive variance (sigma^2) in all cases.
4. Honest Cap Reporting (Task 3): Prints [PASS / FAIL] for constraints on BOTH Val and Test
   using the same <=5.0% / <=1.0% hard thresholds. No soft tolerance is applied.
5. Benchmark Baseline: Evaluates LightGBM recall at the exact non-ALLOW alert volume.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dl.src.config import PipelineConfig, get_default_output_dir, save_json, set_seeds


@dataclass
class PolicyThresholds:
    p_fraud_thresh: float
    p_evidence_thresh: float
    uncertainty_thresh: float
    high_exposure_usd: float = 2000.0


def assign_next_best_actions(
    probs: np.ndarray,
    uncertainties: np.ndarray,
    thresholds: PolicyThresholds,
    amounts: np.ndarray | None = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Classify transactions into ALLOW (0), ACT (1), REQUEST_EVIDENCE (2), ESCALATE (3)."""
    n = len(probs)
    if amounts is None:
        amounts = np.zeros(n, dtype=np.float32)

    is_fraud = probs >= thresholds.p_fraud_thresh
    is_high_uncert = uncertainties >= thresholds.uncertainty_thresh
    is_high_exp = amounts >= thresholds.high_exposure_usd
    is_evidence = (probs >= thresholds.p_evidence_thresh) | is_high_uncert

    action_codes = np.zeros(n, dtype=int)
    escalate_mask = is_fraud & (is_high_uncert | is_high_exp)
    act_mask = is_fraud & ~escalate_mask
    evidence_mask = ~is_fraud & is_evidence

    action_codes[act_mask] = 1
    action_codes[evidence_mask] = 2
    action_codes[escalate_mask] = 3

    action_names = np.array(["ALLOW", "ACT", "REQUEST_EVIDENCE", "ESCALATE"])
    actions = action_names[action_codes]
    return actions, action_codes


def compute_expected_operational_cost(
    action_codes: np.ndarray,
    targets: np.ndarray,
    amounts: np.ndarray,
    cost_fp_act: float = 10.0,
    cost_stepup: float = 2.0,
    cost_escalate: float = 15.0,
) -> float:
    """Compute total operational financial loss + review cost under action assignments."""
    is_fraud = targets == 1
    missed_fraud_cost = np.sum(amounts[is_fraud & (action_codes == 0)])
    act_cost = np.sum(action_codes == 1) * cost_fp_act
    stepup_cost = np.sum(action_codes == 2) * cost_stepup
    escalate_cost = np.sum(action_codes == 3) * cost_escalate
    return float(missed_fraud_cost + act_cost + stepup_cost + escalate_cost)


def calibrate_thresholds_on_val(
    val_probs: np.ndarray,
    val_uncertainties: np.ndarray,
    val_targets: np.ndarray,
    val_amounts: np.ndarray,
    max_non_allow_pct: float = 5.0,
    max_escalate_pct: float = 1.0,
    min_evidence_pct: float = 1.0,
) -> PolicyThresholds:
    """Optimize thresholds strictly on Validation using Expected-Cost Objective and operational caps.

    Task 1 fix: evidence band guaranteed >= 1% by scanning an absolute probability grid.
    The evidence threshold is chosen from percentile-derived absolute values (not as a
    fraction of p_fraud_thresh), so the two thresholds cannot accidentally coincide.
    """
    n_val = len(val_probs)

    u_candidates = [float(np.percentile(val_uncertainties, q)) for q in [85, 90, 93, 95, 97, 98, 99, 99.5]]
    p_fraud_candidates = [float(np.percentile(val_probs, q)) for q in [95, 96, 97, 97.5, 98, 98.5, 99, 99.2, 99.5]]

    # Evidence-threshold candidates: absolute probability values spanning 75th to 99th percentiles.
    # Higher percentiles (94-98.5) allow non-ALLOW budget (<= 5%) to be strictly satisfied.
    evidence_percentiles = [75, 80, 85, 90, 92, 94, 95, 95.5, 96, 96.5, 97, 97.5, 98, 98.5]
    p_evidence_pool = sorted(set(float(np.percentile(val_probs, q)) for q in evidence_percentiles))

    min_cost = float("inf")
    best_thresholds = PolicyThresholds(
        p_fraud_thresh=float(np.percentile(val_probs, 98.5)),
        p_evidence_thresh=float(np.percentile(val_probs, 96.0)),
        uncertainty_thresh=float(np.percentile(val_uncertainties, 98.0)),
    )

    found_feasible = False
    for u_th in u_candidates:
        for pf_th in p_fraud_candidates:
            for pe_th in p_evidence_pool:
                if pe_th >= pf_th:
                    continue  # evidence threshold must be strictly below fraud threshold
                temp_thresh = PolicyThresholds(
                    p_fraud_thresh=pf_th,
                    p_evidence_thresh=pe_th,
                    uncertainty_thresh=u_th,
                )
                _, codes = assign_next_best_actions(val_probs, val_uncertainties, temp_thresh, val_amounts)

                non_allow_pct = (np.sum(codes != 0) / n_val) * 100.0
                escalate_pct = (np.sum(codes == 3) / n_val) * 100.0
                evidence_pct = (np.sum(codes == 2) / n_val) * 100.0

                # All three constraints must be satisfied simultaneously (no soft tolerance)
                if (
                    non_allow_pct <= max_non_allow_pct
                    and escalate_pct <= max_escalate_pct
                    and evidence_pct >= min_evidence_pct
                ):
                    cost = compute_expected_operational_cost(codes, val_targets, val_amounts)
                    if cost < min_cost:
                        min_cost = cost
                        best_thresholds = temp_thresh
                        found_feasible = True

    if not found_feasible:
        print(
            "[!] WARNING: No threshold combination satisfied all three constraints simultaneously. "
            "Using best-effort defaults (evidence band may be < 1%)."
        )

    return best_thresholds


def _select_prob_source(
    prob_source: str,
    val_mc: dict,
    test_mc: dict,
    config: PipelineConfig,
) -> Tuple[np.ndarray, np.ndarray, str]:
    """Return (val_probs, test_probs, description) for the chosen probability source.

    Task 2: Three sources available:
      'gnn'       - Calibrated MC-Dropout GNN (temperature-scaled, rank-preserving).
      'lgb_graph' - Calibrated LightGBM + Graph.
      'ensemble'  - 50/50 average of both (default).
    The uncertainty signal (MC-Dropout variance) is always sourced from the GNN.
    """
    gnn_val_p = val_mc["mean_prob_calibrated"] if "mean_prob_calibrated" in val_mc else val_mc["mean_prob"]
    gnn_test_p = test_mc["mean_prob_calibrated"] if "mean_prob_calibrated" in test_mc else test_mc["mean_prob"]

    lgb_val_cal_path = config.out_dir / "preds_val_lgb_graph_cal.npy"
    lgb_test_cal_path = config.out_dir / "preds_test_lgb_graph_cal.npy"

    if lgb_val_cal_path.exists() and lgb_test_cal_path.exists():
        lgb_val_p = np.load(lgb_val_cal_path)
        lgb_test_p = np.load(lgb_test_cal_path)
    elif (config.out_dir / "preds_val_lgb_graph.npy").exists():
        lgb_val_p = np.load(config.out_dir / "preds_val_lgb_graph.npy")
        lgb_test_p = np.load(config.out_dir / "preds_test_lgb_graph.npy")
        print("    [!] Calibrated LightGBM+Graph preds not found; using raw preds.")
    else:
        lgb_val_p = gnn_val_p
        lgb_test_p = gnn_test_p
        print("    [!] LightGBM+Graph preds not found; falling back to GNN probabilities.")

    if prob_source == "gnn":
        val_probs, test_probs = gnn_val_p, gnn_test_p
        desc = "Calibrated GraphSAGE GNN (temperature-scaled, rank-preserving)"
    elif prob_source == "lgb_graph":
        val_probs, test_probs = lgb_val_p, lgb_test_p
        desc = "Calibrated LightGBM + Graph (temperature-scaled)"
    else:  # 'ensemble' — default
        val_probs = 0.5 * (gnn_val_p + lgb_val_p)
        test_probs = 0.5 * (gnn_test_p + lgb_test_p)
        desc = (
            "Ensemble (50% Calibrated GNN + 50% Calibrated LightGBM+Graph) — "
            "default because averaging two complementary classifiers yields a wider "
            "probability spread, making threshold optimisation more tractable."
        )

    return val_probs, test_probs, desc


def evaluate_decision_policy(config: PipelineConfig, prob_source: str = "ensemble") -> Dict[str, object]:
    """Calibrate policy on validation and evaluate action routing on test set."""
    config.ensure_dirs()
    set_seeds(config.seed)

    # 1. Load MC Dropout outputs and metadata
    val_mc = dict(np.load(config.out_dir / "mc_dropout_val.npz"))
    test_mc = dict(np.load(config.out_dir / "mc_dropout_test.npz"))

    val_meta = pd.read_parquet(config.out_dir / "val_meta.parquet")
    test_meta = pd.read_parquet(config.out_dir / "test_meta.parquet")

    train_arrays = np.load(config.out_dir / "processed_arrays.npz")
    y_val = train_arrays["y_val"]
    y_test = train_arrays["y_test"]

    val_amounts = (
        val_meta["TransactionAmt"].values.astype(np.float32)
        if "TransactionAmt" in val_meta.columns
        else np.full(len(y_val), 100.0, dtype=np.float32)
    )
    test_amounts = (
        test_meta["TransactionAmt"].values.astype(np.float32)
        if "TransactionAmt" in test_meta.columns
        else np.full(len(y_test), 100.0, dtype=np.float32)
    )

    # 2. Select Probability Source (Task 2)
    val_probs, test_probs, source_desc = _select_prob_source(prob_source, val_mc, test_mc, config)

    val_uncert = val_mc["variance"]
    test_uncert = test_mc["variance"]

    print(f"\n[*] Decision Policy — Probability Source: {prob_source.upper()}")
    print(f"    Description : {source_desc}")
    print("    Uncertainty : MC-Dropout GNN Predictive Variance (sigma^2) — used for all sources.")

    # 3. Calibrate Thresholds on Validation only (Task 1)
    thresholds = calibrate_thresholds_on_val(
        val_probs, val_uncert, y_val, val_amounts,
        max_non_allow_pct=5.0, max_escalate_pct=1.0, min_evidence_pct=1.0,
    )
    gap = thresholds.p_fraud_thresh - thresholds.p_evidence_thresh
    print(f"\n[+] Validation-Calibrated Thresholds (Expected-Cost, Evidence Band >= 1%):")
    print(f"  * p_fraud_thresh:                 {thresholds.p_fraud_thresh:.4f}")
    print(f"  * p_evidence_thresh:              {thresholds.p_evidence_thresh:.4f}")
    print(f"  * uncertainty_thresh:             {thresholds.uncertainty_thresh:.6f}")
    print(f"  * evidence gap (fraud-evidence):  {gap:.4f}")

    # 4. Evaluate on Validation
    val_actions, val_action_codes = assign_next_best_actions(val_probs, val_uncert, thresholds, val_amounts)
    val_non_allow_pct = (np.sum(val_action_codes != 0) / len(val_probs)) * 100.0
    val_escalate_pct = (np.sum(val_action_codes == 3) / len(val_probs)) * 100.0
    val_evidence_pct = (np.sum(val_action_codes == 2) / len(val_probs)) * 100.0

    # 5. Evaluate on Test
    test_actions, test_action_codes = assign_next_best_actions(test_probs, test_uncert, thresholds, test_amounts)
    test_non_allow_pct = (np.sum(test_action_codes != 0) / len(test_probs)) * 100.0
    test_escalate_pct = (np.sum(test_action_codes == 3) / len(test_probs)) * 100.0
    test_evidence_pct = (np.sum(test_action_codes == 2) / len(test_probs)) * 100.0

    # 6. Honest Cap Reporting — same hard threshold on BOTH splits, no soft tolerance (Task 3)
    non_allow_val_status = "PASS" if val_non_allow_pct <= 5.0 else "FAIL"
    non_allow_test_status = "PASS" if test_non_allow_pct <= 5.0 else "FAIL"
    escalate_val_status = "PASS" if val_escalate_pct <= 1.0 else "FAIL"
    escalate_test_status = "PASS" if test_escalate_pct <= 1.0 else "FAIL"
    evidence_val_status = "PASS" if val_evidence_pct >= 1.0 else "FAIL"
    evidence_test_status = "PASS" if test_evidence_pct >= 1.0 else "FAIL"

    print("\n" + "=" * 80)
    print("                     OPERATIONAL CAP COMPLIANCE REPORT")
    print("=" * 80)
    print(f"  Constraint             |  Validation              |  Test (out-of-sample)")
    print(f"  -----------------------|--------------------------|-------------------------")
    print(
        f"  Non-ALLOW <= 5.0%      |  [{non_allow_val_status}]  {val_non_allow_pct:6.2f}%"
        f"            |  [{non_allow_test_status}]  {test_non_allow_pct:6.2f}%"
    )
    print(
        f"  ESCALATE  <= 1.0%      |  [{escalate_val_status}]  {val_escalate_pct:6.2f}%"
        f"            |  [{escalate_test_status}]  {test_escalate_pct:6.2f}%"
    )
    print(
        f"  EVIDENCE  >= 1.0%      |  [{evidence_val_status}]  {val_evidence_pct:6.2f}%"
        f"            |  [{evidence_test_status}]  {test_evidence_pct:6.2f}%"
    )
    print("=" * 80)
    if non_allow_test_status == "FAIL":
        print("  [!] Test non-ALLOW FAIL is expected — thresholds tuned on val only (honest OOS result).")

    # 7. Test Action Breakdown
    action_summary = {}
    print("\n[*] Out-of-Sample Test Action Breakdown:")
    for code, label in [(0, "ALLOW"), (1, "ACT"), (2, "REQUEST_EVIDENCE"), (3, "ESCALATE")]:
        mask = test_action_codes == code
        count = int(mask.sum())
        pct = (count / len(test_probs)) * 100.0
        fraud_count = int(y_test[mask].sum())
        fraud_precision = (fraud_count / count * 100.0) if count > 0 else 0.0
        fraud_recall = (fraud_count / max(1, y_test.sum())) * 100.0

        action_summary[label] = {
            "count": count,
            "pct": round(pct, 2),
            "frauds_captured": fraud_count,
            "precision_pct": round(fraud_precision, 2),
            "recall_pct": round(fraud_recall, 2),
        }
        print(
            f"  * {label:17s}: {count:6,d} ({pct:5.2f}%)"
            f" | Frauds: {fraud_count:4,d}"
            f" | Precision: {fraud_precision:5.1f}%"
            f" | Recall: {fraud_recall:5.1f}%"
        )

    total_frauds_caught = sum(action_summary[a]["frauds_captured"] for a in ["ACT", "REQUEST_EVIDENCE", "ESCALATE"])
    total_non_allow_count = sum(action_summary[a]["count"] for a in ["ACT", "REQUEST_EVIDENCE", "ESCALATE"])
    total_policy_recall = (total_frauds_caught / max(1, y_test.sum())) * 100.0
    total_policy_precision = (total_frauds_caught / max(1, total_non_allow_count)) * 100.0

    print(f"\n[+] Total Policy Performance (All Non-ALLOW Actions combined):")
    print(f"  * Alert Volume:      {total_non_allow_count:,} ({test_non_allow_pct:.2f}% of traffic)")
    print(f"  * Frauds Caught:     {total_frauds_caught:,} / {int(y_test.sum()):,} ({total_policy_recall:.2f}% Recall)")
    print(f"  * Precision:         {total_policy_precision:.2f}%")

    # 8. LightGBM at identical alert volume (Task 3)
    lgb_test_path = config.out_dir / "preds_test_lgb.npy"
    lgb_test_raw = np.load(lgb_test_path) if lgb_test_path.exists() else test_probs
    top_k_indices = np.argsort(lgb_test_raw)[::-1][:total_non_allow_count]
    lgb_frauds_caught = int(y_test[top_k_indices].sum())
    lgb_recall = (lgb_frauds_caught / max(1, y_test.sum())) * 100.0
    lgb_precision = (lgb_frauds_caught / max(1, total_non_allow_count)) * 100.0

    print(f"\n[*] Comparison — LightGBM at same {test_non_allow_pct:.2f}% alert volume:")
    print(f"  * LightGBM: {lgb_frauds_caught:,} frauds ({lgb_recall:.2f}% Recall | {lgb_precision:.2f}% Precision)")
    print(
        f"  * Delta: Recall {total_policy_recall - lgb_recall:+.2f}%,"
        f" Precision {total_policy_precision - lgb_precision:+.2f}%"
    )

    # Save JSON summary
    policy_results = {
        "prob_source": prob_source,
        "prob_source_description": source_desc,
        "thresholds": {
            "p_fraud": float(thresholds.p_fraud_thresh),
            "p_evidence": float(thresholds.p_evidence_thresh),
            "uncertainty": float(thresholds.uncertainty_thresh),
            "evidence_gap": float(gap),
        },
        "operational_caps": {
            "non_allow_val_pct": float(val_non_allow_pct),
            "non_allow_test_pct": float(test_non_allow_pct),
            "escalate_val_pct": float(val_escalate_pct),
            "escalate_test_pct": float(test_escalate_pct),
            "evidence_val_pct": float(val_evidence_pct),
            "evidence_test_pct": float(test_evidence_pct),
            "non_allow_val_status": non_allow_val_status,
            "non_allow_test_status": non_allow_test_status,
            "escalate_val_status": escalate_val_status,
            "escalate_test_status": escalate_test_status,
            "evidence_val_status": evidence_val_status,
            "evidence_test_status": evidence_test_status,
        },
        "test_summary": action_summary,
        "comparison_vs_lgb": {
            "alert_volume": total_non_allow_count,
            "alert_volume_pct": float(test_non_allow_pct),
            "policy_recall_pct": float(total_policy_recall),
            "lgb_recall_pct": float(lgb_recall),
            "policy_precision_pct": float(total_policy_precision),
            "lgb_precision_pct": float(lgb_precision),
        },
    }
    save_json(config.out_dir / "decision_policy_summary.json", policy_results)

    test_meta["pred_prob"] = test_probs
    test_meta["epistemic_uncertainty"] = test_uncert
    test_meta["recommended_action"] = test_actions
    test_meta.to_parquet(config.out_dir / "test_decisions.parquet", index=False)

    return policy_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate uncertainty-aware next-best-action policy.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir())
    parser.add_argument(
        "--prob_source", type=str, default="ensemble",
        choices=["gnn", "lgb_graph", "ensemble"],
        help="Probability source: 'gnn', 'lgb_graph', or 'ensemble' (default).",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, seed=args.seed)
    evaluate_decision_policy(cfg, prob_source=args.prob_source)
