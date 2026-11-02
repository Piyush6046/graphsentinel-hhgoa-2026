"""Probability Calibration via Temperature Scaling (default, rank-preserving) and Isotonic Regression.

Task 6 changes:
- Temperature Scaling is now the DEFAULT method (rank-preserving, does not alter ROC-AUC).
- Isotonic regression is kept as an option via --method isotonic.
- ALL models are calibrated: LightGBM, XGBoost, Tabular MLP, LightGBM+Graph, GraphSAGE (GNN),
  GNN+AE, and MC-Dropout GNN.
- Test ECE is reported for all models both before and after calibration.
- A unified calibration summary table is printed.

Fits calibrators STRICTLY on the VALIDATION split to correct probability distributions
and reduce Expected Calibration Error (ECE) across all deep learning and tree models.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar
from scipy.special import expit, logit
from sklearn.isotonic import IsotonicRegression

from dl.src.config import PipelineConfig, get_default_output_dir, save_json, set_seeds
from dl.src.evaluate import compute_expected_calibration_error


class TemperatureScaler:
    """Rank-preserving Temperature Scaling for probability calibration.

    Minimizes negative log-likelihood on the validation set with respect to a single
    scalar temperature T > 0.  The ranking of predictions is strictly preserved.
    """

    def __init__(self):
        self.temperature: float = 1.0
        self.is_fitted: bool = False

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "TemperatureScaler":
        eps = 1e-6
        p_clamped = np.clip(val_probs, eps, 1.0 - eps)
        z = logit(p_clamped)

        def nll_loss(t: float) -> float:
            if t <= 0.0:
                return 1e9
            scaled_p = expit(z / t)
            scaled_p = np.clip(scaled_p, 1e-7, 1.0 - 1e-7)
            bce = -np.mean(y_val * np.log(scaled_p) + (1.0 - y_val) * np.log(1.0 - scaled_p))
            return float(bce)

        res = minimize_scalar(nll_loss, bounds=(0.05, 10.0), method="bounded")
        self.temperature = float(res.x)
        self.is_fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("TemperatureScaler must be fitted before transforming.")
        eps = 1e-6
        p_clamped = np.clip(probs, eps, 1.0 - eps)
        z = logit(p_clamped)
        calibrated = expit(z / self.temperature)
        return np.clip(calibrated, 1e-5, 1.0 - 1e-5)


class IsotonicCalibrator:
    """Non-parametric monotonic Isotonic Regression calibrator."""

    def __init__(self):
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=1e-5, y_max=1.0 - 1e-5)
        self.is_fitted: bool = False

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "IsotonicCalibrator":
        self.calibrator.fit(val_probs, y_val)
        self.is_fitted = True
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("IsotonicCalibrator must be fitted before transforming.")
        return np.clip(self.calibrator.predict(probs), 1e-5, 1.0 - 1e-5)


class ProbabilityCalibrator:
    """Backwards-compatible alias wrapping TemperatureScaler (used in tests)."""

    def __init__(self, method: str = "temp_scaling"):
        self._inner = TemperatureScaler() if method == "temp_scaling" else IsotonicCalibrator()

    def fit(self, val_probs: np.ndarray, y_val: np.ndarray) -> "ProbabilityCalibrator":
        self._inner.fit(val_probs, y_val)
        return self

    def transform(self, probs: np.ndarray) -> np.ndarray:
        return self._inner.transform(probs)

    @property
    def temperature(self):
        return getattr(self._inner, "temperature", None)


def get_calibrator(method: str = "temp_scaling"):
    """Factory: return a TemperatureScaler or IsotonicCalibrator."""
    if method == "temp_scaling":
        return TemperatureScaler()
    elif method == "isotonic":
        return IsotonicCalibrator()
    raise ValueError(f"Unknown calibration method: '{method}'. Choose 'temp_scaling' (default) or 'isotonic'.")


def calibrate_predictions(config: PipelineConfig, method: str = "temp_scaling") -> Dict[str, Dict[str, float]]:
    """Calibrate all models using validation data and persist calibrated test predictions.

    Task 6: Temperature Scaling is the default (rank-preserving).
    All models (tabular baselines + GNN + MC-Dropout) are calibrated.
    A unified ECE table is printed for all models.
    """
    config.ensure_dirs()
    set_seeds(config.seed)

    val_meta = pd.read_parquet(config.out_dir / "val_meta.parquet")
    test_meta = pd.read_parquet(config.out_dir / "test_meta.parquet")

    y_val = val_meta["isFraud"].values.astype(np.float32)
    y_test = test_meta["isFraud"].values.astype(np.float32)

    # All tabular/GNN model pairs: (display name, val_file, test_file)
    model_files = {
        "LightGBM": ("preds_val_lgb.npy", "preds_test_lgb.npy"),
        "XGBoost": ("preds_val_xgb.npy", "preds_test_xgb.npy"),
        "Tabular MLP": ("preds_val_mlp.npy", "preds_test_mlp.npy"),
        "LightGBM + Graph": ("preds_val_lgb_graph.npy", "preds_test_lgb_graph.npy"),
        "GraphSAGE (GNN)": ("preds_val_gnn.npy", "preds_test_gnn.npy"),
        "GNN + AE (Full)": ("preds_val_gnn_ae.npy", "preds_test_gnn_ae.npy"),
    }

    ece_results = {}
    print(f"\n[*] Calibrating all models (Method: {method.upper()}, fitted on VALIDATION split only)...")
    print(f"\n  {'Model':24s}  {'ECE Before':>11}  {'ECE After':>10}  {'T':>6}")
    print(f"  {'-'*24}  {'-'*11}  {'-'*10}  {'-'*6}")

    for name, (val_file, test_file) in model_files.items():
        v_path = config.out_dir / val_file
        t_path = config.out_dir / test_file
        if not (v_path.exists() and t_path.exists()):
            continue

        val_p = np.load(v_path)
        test_p = np.load(t_path)

        ece_before, _, _, _ = compute_expected_calibration_error(y_test, test_p)

        calibrator = get_calibrator(method).fit(val_p, y_val)
        cal_val = calibrator.transform(val_p)
        cal_test = calibrator.transform(test_p)

        ece_after, _, _, _ = compute_expected_calibration_error(y_test, cal_test)

        # Save calibrated predictions (suffix _cal.npy)
        cal_test_path = config.out_dir / test_file.replace(".npy", "_cal.npy")
        cal_val_path = config.out_dir / val_file.replace(".npy", "_cal.npy")
        np.save(cal_test_path, cal_test)
        np.save(cal_val_path, cal_val)

        t_str = f"{calibrator.temperature:.3f}" if hasattr(calibrator, "temperature") else "n/a"
        print(f"  {name:24s}  {ece_before:11.4f}  {ece_after:10.4f}  {t_str:>6}")

        ece_results[name] = {
            "test_ece_raw": float(ece_before),
            "test_ece_calibrated": float(ece_after),
            "temperature": float(calibrator.temperature) if hasattr(calibrator, "temperature") else None,
        }

    # Calibrate MC-Dropout GNN
    mc_val_path = config.out_dir / "mc_dropout_val.npz"
    mc_test_path = config.out_dir / "mc_dropout_test.npz"

    if mc_val_path.exists() and mc_test_path.exists():
        val_mc = dict(np.load(mc_val_path))
        test_mc = dict(np.load(mc_test_path))

        raw_val_p = val_mc["mean_prob"]
        raw_test_p = test_mc["mean_prob"]

        ece_mc_before, _, _, _ = compute_expected_calibration_error(y_test, raw_test_p)

        mc_calibrator = get_calibrator(method).fit(raw_val_p, y_val)
        cal_mc_val = mc_calibrator.transform(raw_val_p)
        cal_mc_test = mc_calibrator.transform(raw_test_p)

        ece_mc_after, _, _, _ = compute_expected_calibration_error(y_test, cal_mc_test)

        # Recalculate calibrated entropy
        eps = 1e-7
        p_c = np.clip(cal_mc_test, eps, 1.0 - eps)
        cal_entropy = -(p_c * np.log2(p_c) + (1.0 - p_c) * np.log2(1.0 - p_c))

        t_str = f"{mc_calibrator.temperature:.3f}" if hasattr(mc_calibrator, "temperature") else "n/a"
        print(f"  {'MC-Dropout GNN':24s}  {ece_mc_before:11.4f}  {ece_mc_after:10.4f}  {t_str:>6}")
        print(
            f"    Entropy: {np.mean(test_mc['total_entropy']):.4f} bits (raw) "
            f"-> {np.mean(cal_entropy):.4f} bits (cal)"
        )

        val_mc["mean_prob_calibrated"] = cal_mc_val
        test_mc["mean_prob_calibrated"] = cal_mc_test
        test_mc["total_entropy_calibrated"] = cal_entropy

        np.savez_compressed(mc_val_path, **val_mc)
        np.savez_compressed(mc_test_path, **test_mc)

        ece_results["MC-Dropout GNN"] = {
            "test_ece_raw": float(ece_mc_before),
            "test_ece_calibrated": float(ece_mc_after),
            "temperature": float(mc_calibrator.temperature) if hasattr(mc_calibrator, "temperature") else None,
            "test_mean_entropy_raw": float(np.mean(test_mc["total_entropy"])),
            "test_mean_entropy_calibrated": float(np.mean(cal_entropy)),
        }

    print(f"  {'-'*24}  {'-'*11}  {'-'*10}  {'-'*6}")

    save_json(config.out_dir / "calibration_summary.json", {
        "method": method,
        "models": ece_results,
    })

    print(f"\n[+] Calibrated predictions saved to {config.out_dir} (suffix: _cal.npy / mc_dropout_*.npz)")
    return ece_results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Calibrate all model predictions using validation set.")
    parser.add_argument("--out_dir", type=Path, default=get_default_output_dir())
    parser.add_argument(
        "--method", type=str, default="temp_scaling",
        choices=["temp_scaling", "isotonic"],
        help="Calibration method: 'temp_scaling' (default, rank-preserving) or 'isotonic'.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PipelineConfig(out_dir=args.out_dir, seed=args.seed)
    calibrate_predictions(cfg, method=args.method)
