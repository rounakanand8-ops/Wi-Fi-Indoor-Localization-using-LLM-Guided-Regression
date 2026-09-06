"""Evaluation metrics for the (X, Y) localization task."""

from __future__ import annotations

import numpy as np


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """
    y_true, y_pred: shape (n, 2), columns = [X, Y]

    Returns per-axis MAE/MSE (as explicitly requested) plus the mean/median
    Euclidean positioning error in meters, which is the metric the published
    state-of-the-art numbers (1.85 / 1.23 / 1.98 m) refer to.
    """
    err = y_pred - y_true
    abs_err = np.abs(err)
    sq_err = err ** 2

    euclidean = np.sqrt(np.sum(sq_err, axis=1))

    return {
        "MAE_X": float(abs_err[:, 0].mean()),
        "MAE_Y": float(abs_err[:, 1].mean()),
        "MSE_X": float(sq_err[:, 0].mean()),
        "MSE_Y": float(sq_err[:, 1].mean()),
        "MAE_overall": float(abs_err.mean()),
        "MSE_overall": float(sq_err.mean()),
        "mean_euclidean_error_m": float(euclidean.mean()),
        "median_euclidean_error_m": float(np.median(euclidean)),
        "p90_euclidean_error_m": float(np.percentile(euclidean, 90)),
    }


def print_report(env_name: str, metrics: dict, sota_m: float) -> None:
    beat = metrics["mean_euclidean_error_m"] < sota_m
    print(f"\n=== {env_name} ===")
    print(f"  MAE  (X, Y): {metrics['MAE_X']:.3f} m, {metrics['MAE_Y']:.3f} m")
    print(f"  MSE  (X, Y): {metrics['MSE_X']:.3f}, {metrics['MSE_Y']:.3f}")
    print(f"  Mean Euclidean error : {metrics['mean_euclidean_error_m']:.3f} m")
    print(f"  Median Euclidean error: {metrics['median_euclidean_error_m']:.3f} m")
    print(f"  90th pct error       : {metrics['p90_euclidean_error_m']:.3f} m")
    print(f"  Published SOTA       : {sota_m:.2f} m")
    print(f"  Result: {'BEATS SOTA ✅' if beat else 'does NOT beat SOTA ❌'}")
