"""Uncertainty-Driven Next-Best-Action Policy Engine for Automated Fraud Triage.

The decision engine routes transactions into 4 distinct actions based on the 2D plane
of (Fraud Probability p, Epistemic Uncertainty sigma^2):
1. ALLOW: Confident Legitimate (Low p, Low sigma^2)
2. ACT (Auto-Block / Decline): Confident Fraud (High p, Low/Moderate sigma^2)
3. REQUEST_EVIDENCE (Step-Up Auth / OTP): Ambiguous / High Uncertainty Mid-Risk
4. ESCALATE (Analyst Review / SAR Filing): High Fraud Risk + High Uncertainty or High Exposure.

All decision thresholds are tuned STRICTLY on the VALIDATION split.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd

from dl.src.config import PipelineConfig, get_default_output_dir, set_seeds


@dataclass
class PolicyThresholds:
    p_fraud_thresh: float
    p_evidence_thresh: float
    uncertainty_thresh: float
    high_exposure_usd: float = 2000.0


def calibrate_thresholds_on_val(
    val_probs: np.ndarray,
    val_uncertainties: np.ndarray,
    val_targets: np.ndarray,
    alert_budget: float = 0.01,
) -> PolicyThresholds:
    """Optimize decision thresholds on validation split under fixed alert capacity."""
    # Find probability threshold for top alert_budget fraction
    p_fraud_thresh = float(np.percentile(val_probs, 100 * (1.0 - alert_budget)))

    # Set evidence request threshold at 80th percentile of positive predictions
    p_evidence_thresh = float(p_fraud_thresh * 0.5)

    # Uncertainty threshold: 85th percentile of validation epistemic variance
    uncert_thresh = float(np.percentile(val_uncertainties, 85.0))

    return PolicyThresholds(
        p_fraud_thresh=p_fraud_thresh,
        p_evidence_thresh=p_evidence_thresh,
        uncertainty_thresh=uncert_thresh,
    )


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

    val_probs = val_mc["mean_prob"]
    val_uncert = val_mc["variance"]

    test_probs = test_mc["mean_prob"]
    test_uncert = test_mc["variance"]

    print("[*] Calibrating Next-Best-Action thresholds strictly on Validation split...")
    thresholds = calibrate_thresholds_on_val(
        val_probs, val_uncert, y_val, alert_budget=config.alert_budget_fraction
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
        pct = (count / len(test_probs)) * 100
        fraud_count = int(y_test[mask].sum())
        fraud_precision = (fraud_count / count * 100) if count > 0 else 0.0
        fraud_recall = (fraud_count / max(1, y_test.sum())) * 100

        action_summary[label] = {
            "count": count,
            "pct": pct,
            "frauds_captured": fraud_count,
            "precision_pct": fraud_precision,
            "recall_pct": fraud_recall,
        }
        print(f"  • {label:17s}: {count:6,d} ({pct:5.2f}%) | Frauds: {fraud_count:4,d} | Precision: {fraud_precision:5.1f}% | Recall: {fraud_recall:5.1f}%")

    # Save decisions
    test_meta["pred_prob"] = test_probs
    test_meta["epistemic_uncertainty"] = test_uncert
    test_meta["total_entropy"] = test_mc["total_entropy"]
    test_meta["recommended_action"] = test_actions
    test_meta.to_parquet(config.out_dir / "test_decisions.parquet", index=False)

    policy_meta = {
        "thresholds": {
            "p_fraud": thresholds.p_fraud_thresh,
            "p_evidence": thresholds.p_evidence_thresh,
            "uncertainty": thresholds.uncertainty_thresh,
        },
        "test_summary": action_summary,
    }
    with open(config.out_dir / "decision_policy_summary.json", "w") as f:
        json.dump(policy_meta, f, indent=2)

    print(f"[+] Decision policy evaluation complete. Saved to {config.out_dir}")
    return action_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate uncertainty-aware next-best-action policy.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, seed=args.seed)
    evaluate_decision_policy(cfg)
