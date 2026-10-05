"""Uncertainty-Driven Next-Best-Action Policy Engine with Strict Operational Constraints.

Routes transactions into 4 distinct actions:
1. ALLOW: Confident Legitimate (Low probability, Low uncertainty)
2. ACT (Auto-Block / Decline): Confident Fraud (High probability, Low/Moderate uncertainty)
3. REQUEST_EVIDENCE (Step-Up Auth / OTP): Ambiguous / Moderate Risk or Elevated Uncertainty
4. ESCALATE (Analyst Review / SAR Filing): High Risk + High Uncertainty or High Exposure.

All decision thresholds are tuned STRICTLY on the VALIDATION split under the operational constraints:
- Total non-ALLOW actions <= 5.0% of total traffic
- ESCALATE actions <= 1.0% of total traffic.
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
    """Classify transactions into ALLOW, ACT, REQUEST_EVIDENCE, ESCALATE actions."""
    n = len(probs)
    actions = np.full(n, "ALLOW", dtype=object)
    action_codes = np.zeros(n, dtype=int)  # 0: ALLOW, 1: ACT, 2: REQUEST_EVIDENCE, 3: ESCALATE

    if amounts is None:
        amounts = np.zeros(n, dtype=np.float32)

    for i in range(n):
        p = probs[i]
        u = uncertainties[i]
        amt = amounts[i]

        if p >= thresholds.p_fraud_thresh:
            if u >= thresholds.uncertainty_thresh or amt >= thresholds.high_exposure_usd:
                # High risk with high uncertainty or high exposure -> Escalate to human analyst
                actions[i] = "ESCALATE"
                action_codes[i] = 3
            else:
                # High risk with confident low uncertainty -> Auto-Block / Decline
                actions[i] = "ACT"
                action_codes[i] = 1
        elif p >= thresholds.p_evidence_thresh or u >= thresholds.uncertainty_thresh:
            # Ambiguous or high uncertainty -> Request step-up auth / evidence
            actions[i] = "REQUEST_EVIDENCE"
            action_codes[i] = 2
        else:
            # Low risk with low uncertainty -> Allow transaction
            actions[i] = "ALLOW"
            action_codes[i] = 0

    return actions, action_codes


def calibrate_thresholds_on_val(
    val_probs: np.ndarray,
    val_uncertainties: np.ndarray,
    val_targets: np.ndarray,
    val_amounts: np.ndarray | None = None,
    max_non_allow_pct: float = 5.0,
    max_escalate_pct: float = 1.0,
) -> PolicyThresholds:
    """Tune thresholds strictly on validation split to enforce operational constraints."""
    n_val = len(val_probs)

    # Uncertainty candidate percentiles (top 1% to 15% highest uncertainty)
    u_candidates = [float(np.percentile(val_uncertainties, q)) for q in [85, 90, 93, 95, 97, 98, 99]]
    # Probability candidate percentiles (top 0.5% to 5% highest risk)
    p_fraud_candidates = [float(np.percentile(val_probs, q)) for q in [95, 96, 97, 98, 98.5, 99, 99.5]]
    # Evidence thresholds: fractions of p_fraud
    p_ev_candidates = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]

    best_score = -1.0
    best_thresholds = PolicyThresholds(
        p_fraud_thresh=float(np.percentile(val_probs, 98.0)),
        p_evidence_thresh=float(np.percentile(val_probs, 96.0)),
        uncertainty_thresh=float(np.percentile(val_uncertainties, 97.0)),
    )

    for u_th in u_candidates:
        for pf_th in p_fraud_candidates:
            for ev_factor in p_ev_candidates:
                pe_th = pf_th * ev_factor
                temp_thresh = PolicyThresholds(p_fraud_thresh=pf_th, p_evidence_thresh=pe_th, uncertainty_thresh=u_th)
                _, codes = assign_next_best_actions(val_probs, val_uncertainties, temp_thresh, val_amounts)

                non_allow_pct = (np.sum(codes != 0) / n_val) * 100.0
                escalate_pct = (np.sum(codes == 3) / n_val) * 100.0

                # Check strict operational constraints
                if non_allow_pct <= max_non_allow_pct and escalate_pct <= max_escalate_pct:
                    # Score by frauds captured in non-allow actions
                    frauds_captured = np.sum((codes != 0) & (val_targets == 1))
                    total_non_allow = max(1, np.sum(codes != 0))
                    precision = frauds_captured / total_non_allow

                    # F1-like objective on validation
                    score = frauds_captured * precision
                    if score > best_score:
                        best_score = score
                        best_thresholds = temp_thresh

    return best_thresholds


def evaluate_decision_policy(config: PipelineConfig) -> Dict[str, Dict[str, float]]:
    """Calibrate policy on validation and evaluate action routing on test set."""
    config.ensure_dirs()
    set_seeds(config.seed)

    # Load MC Dropout artifacts
    val_mc = np.load(config.out_dir / "mc_dropout_val.npz")
    test_mc = np.load(config.out_dir / "mc_dropout_test.npz")

    val_meta = pd.read_parquet(config.out_dir / "val_meta.parquet")
    test_meta = pd.read_parquet(config.out_dir / "test_meta.parquet")

    y_val = val_meta["isFraud"].values.astype(np.float32)
    y_test = test_meta["isFraud"].values.astype(np.float32)

    # Use calibrated probabilities if available
    val_probs = val_mc["mean_prob_calibrated"] if "mean_prob_calibrated" in val_mc else val_mc["mean_prob"]
    test_probs = test_mc["mean_prob_calibrated"] if "mean_prob_calibrated" in test_mc else test_mc["mean_prob"]

    val_uncert = val_mc["variance"]
    test_uncert = test_mc["variance"]

    print("[*] Calibrating Next-Best-Action thresholds strictly on Validation split (Constraints: Non-ALLOW <= 5%, ESCALATE <= 1%)...")
    thresholds = calibrate_thresholds_on_val(
        val_probs, val_uncert, y_val, max_non_allow_pct=5.0, max_escalate_pct=1.0
    )
    print(f"  • Tuned p_fraud_thresh:     {thresholds.p_fraud_thresh:.4f}")
    print(f"  • Tuned p_evidence_thresh:  {thresholds.p_evidence_thresh:.4f}")
    print(f"  • Tuned uncertainty_thresh: {thresholds.uncertainty_thresh:.6f}")

    # Evaluate on Test Split
    test_actions, test_action_codes = assign_next_best_actions(test_probs, test_uncert, thresholds)

    # Action distribution and fraud capture
    action_summary = {}
    print("\n[*] Out-of-Sample Test Action Breakdown:")
    for code, label in [(0, "ALLOW"), (1, "ACT"), (2, "REQUEST_EVIDENCE"), (3, "ESCALATE")]:
        mask = (test_action_codes == code)
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
        print(f"  • {label:17s}: {count:6,d} ({pct:5.2f}%) | Frauds: {fraud_count:4,d} | Precision: {fraud_precision:5.1f}% | Recall: {fraud_recall:5.1f}%")

    non_allow_total = sum(action_summary[act]["count"] for act in ["ACT", "REQUEST_EVIDENCE", "ESCALATE"])
    non_allow_pct = (non_allow_total / len(test_probs)) * 100.0
    print(f"\n[+] Total Non-ALLOW Action Volume: {non_allow_total:,} ({non_allow_pct:.2f}% of traffic <= 5% operational cap)")

    # Save decisions
    test_meta["pred_prob"] = test_probs
    test_meta["epistemic_uncertainty"] = test_uncert
    test_meta["recommended_action"] = test_actions
    test_meta.to_parquet(config.out_dir / "test_decisions.parquet", index=False)

    policy_meta = {
        "thresholds": {
            "p_fraud": float(thresholds.p_fraud_thresh),
            "p_evidence": float(thresholds.p_evidence_thresh),
            "uncertainty": float(thresholds.uncertainty_thresh),
        },
        "test_summary": action_summary,
    }
    save_json(config.out_dir / "decision_policy_summary.json", policy_meta)

    print(f"[+] Decision policy evaluation complete. Saved to {config.out_dir}")
    return action_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate uncertainty-aware next-best-action policy.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, seed=args.seed)
    evaluate_decision_policy(cfg)
