"""Post-Hoc Probability Calibration via Isotonic Regression and Temperature Scaling.

Fits probability calibrators STRICTLY on the VALIDATION split to correct probability
and entropy distortions introduced by focal loss or deep network overconfidence.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.isotonic import IsotonicRegression

from dl.src.config import PipelineConfig, get_default_output_dir, save_json, set_seeds
from dl.src.evaluate import compute_expected_calibration_error


class ProbabilityCalibrator:
    """Isotonic regression calibrator fitted on validation predictions."""

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1.0 - 1e-5)
        self.is_fitted = False

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        self.calibrator.fit(val_probs, y_val)
        self.is_fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("ProbabilityCalibrator must be fitted before transforming.")
        return np.clip(self.calibrator.predict(probs), 1e-5, 1.0 - 1e-5)


def calibrate_predictions(config: PipelineConfig) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Fit calibrator on validation split and update MC Dropout predictions."""
    config.ensure_dirs()
    set_seeds(config.seed)

    # Load MC Dropout artifacts
    val_mc = dict(np.load(config.out_dir / "mc_dropout_val.npz"))
    test_mc = dict(np.load(config.out_dir / "mc_dropout_test.npz"))

    import pandas as pd
    val_meta = pd.read_parquet(config.out_dir / "val_meta.parquet")
    test_meta = pd.read_parquet(config.out_dir / "test_meta.parquet")

    y_val = val_meta["isFraud"].values.astype(np.float32)
    y_test = test_meta["isFraud"].values.astype(np.float32)

    raw_val_probs = val_mc["mean_prob"]
    raw_test_probs = test_mc["mean_prob"]

    # Compute pre-calibration ECE
    ece_val_before, _, _, _ = compute_expected_calibration_error(y_val, raw_val_probs)
    ece_test_before, _, _, _ = compute_expected_calibration_error(y_test, raw_test_probs)

    print(f"[*] Pre-Calibration ECE -> Val: {ece_val_before:.4f} | Test: {ece_test_before:.4f}")

    # Fit calibrator strictly on validation
    calibrator = ProbabilityCalibrator().fit(raw_val_probs, y_val)

    cal_val_probs = calibrator.transform(raw_val_probs)
    cal_test_probs = calibrator.transform(raw_test_probs)

    # Compute post-calibration ECE
    ece_val_after, _, _, _ = compute_expected_calibration_error(y_val, cal_val_probs)
    ece_test_after, _, _, _ = compute_expected_calibration_error(y_test, cal_test_probs)

    print(f"[+] Post-Calibration ECE -> Val: {ece_val_after:.4f} | Test: {ece_test_after:.4f}")

    # Recalculate entropy with calibrated probabilities
    eps = 1e-7
    p_clamped = np.clip(cal_test_probs, eps, 1.0 - eps)
    cal_entropy = -(p_clamped * np.log2(p_clamped) + (1.0 - p_clamped) * np.log2(1.0 - p_clamped))
    print(f"  • Test Mean Calibrated Shannon Entropy: {np.mean(cal_entropy):.4f} bits")

    # Update MC dropout npz files with calibrated probabilities
    val_mc["mean_prob_calibrated"] = cal_val_probs
    test_mc["mean_prob_calibrated"] = cal_test_probs
    test_mc["total_entropy_calibrated"] = cal_entropy

    np.savez_compressed(config.out_dir / "mc_dropout_val.npz", **val_mc)
    np.savez_compressed(config.out_dir / "mc_dropout_test.npz", **test_mc)

    cal_summary = {
        "ece_val_before": float(ece_val_before),
        "ece_val_after": float(ece_val_after),
        "ece_test_before": float(ece_test_before),
        "ece_test_after": float(ece_test_after),
        "test_mean_entropy_before": float(np.mean(test_mc["total_entropy"])),
        "test_mean_entropy_after": float(np.mean(cal_entropy)),
    }
    save_json(config.out_dir / "calibration_summary.json", cal_summary)

    return cal_val_probs, cal_test_probs, cal_summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate model probabilities using validation set.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir(), help="Output artifacts directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, seed=args.seed)
    calibrate_predictions(cfg)
