"""
Train + evaluate a location-prediction model on one or all three
environments (corridor / lecture_theatre / office).

Examples
--------
# Primary path (run locally, after `hf auth login`):
python train.py --model tabpfn --env all

# Smoke-test the pipeline without needing TabPFN's gated weights:
python train.py --model rf --env all
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from data_utils import ENVIRONMENTS, SOTA_MEAN_ERROR_M, build_features
from evaluate import compute_metrics, print_report
from models import build_model


def run_one_environment(env_name: str, data_dir: Path, model_name: str, device: str) -> dict:
    data = build_features(env_name, data_dir)

    model = build_model(model_name, device=device)

    t0 = time.time()
    model.fit(data.X_train, data.y_train)
    fit_s = time.time() - t0

    t0 = time.time()
    y_pred = model.predict(data.X_test)
    predict_s = time.time() - t0

    metrics = compute_metrics(data.y_test, y_pred)
    metrics["fit_seconds"] = fit_s
    metrics["predict_seconds"] = predict_s
    metrics["n_train"] = len(data.y_train)
    metrics["n_test"] = len(data.y_test)

    print_report(env_name, metrics, SOTA_MEAN_ERROR_M[env_name])
    print(f"  (fit {fit_s:.1f}s, predict {predict_s:.1f}s, "
          f"n_train={metrics['n_train']}, n_test={metrics['n_test']})")

    return metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model", default="tabpfn", choices=["tabpfn", "rf", "mlp"],
        help="tabpfn = primary model. rf/mlp = local smoke-test baselines.",
    )
    parser.add_argument(
        "--env", default="all", choices=["all", *ENVIRONMENTS.keys()],
    )
    parser.add_argument("--device", default="auto", help="cpu / cuda / auto (tabpfn only)")
    parser.add_argument("--data-dir", default=str(Path(__file__).parent / "data"))
    args = parser.parse_args()

    envs = list(ENVIRONMENTS.keys()) if args.env == "all" else [args.env]

    all_metrics = {}
    for env_name in envs:
        all_metrics[env_name] = run_one_environment(
            env_name, Path(args.data_dir), args.model, args.device
        )

    if len(envs) > 1:
        print("\n=== Summary ===")
        beats_all = True
        for env_name, m in all_metrics.items():
            sota = SOTA_MEAN_ERROR_M[env_name]
            beat = m["mean_euclidean_error_m"] < sota
            beats_all &= beat
            print(f"  {env_name:16s}: {m['mean_euclidean_error_m']:.3f} m "
                  f"(SOTA {sota:.2f} m) {'✅' if beat else '❌'}")
        print(f"\n  Beats SOTA on all three environments: {beats_all}")


if __name__ == "__main__":
    main()
