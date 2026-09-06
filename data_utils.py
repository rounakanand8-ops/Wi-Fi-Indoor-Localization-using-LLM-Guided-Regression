"""
Data loading + preprocessing for the WiFi RTT/RSS indoor localization dataset.

Each environment has a train/test CSV with columns:
    X, Y, AP1..AP5 RTT(mm), AP1..AP5 RSS(dBm), LOS APs

Sentinel handling
------------------
When an AP is not detected, RTT is recorded as exactly 100000 (mm) and RSS as
exactly -200 (dBm), always together. Left as raw values, this creates a huge
discontinuous jump relative to real RTT/RSS ranges (tens to a few thousand mm,
-40 to -75 dBm), which hurts distance-based / in-context models like TabPFN.

We instead:
  1. Fit per-AP "detected" flags and per-AP cap values from the TRAIN split
     only (never leak test statistics into fitting).
  2. Replace sentinel RTT with (max valid RTT for that AP * 1.2).
  3. Replace sentinel RSS with (min valid RSS for that AP - 5).
  4. Add a binary `AP{i}_detected` feature so the model can still distinguish
     "capped because undetected" from "genuinely near the cap".
  5. Drop any AP column that is constant (always undetected) in train, e.g.
     AP1 in the corridor environment is undetected in 100% of rows.

The resulting feature matrix is fully numeric and identical in shape for
train/test.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

RTT_SENTINEL = 100000.0
RSS_SENTINEL = -200.0
N_APS = 5

ENVIRONMENTS = {
    "corridor": "database_corridor",
    "lecture_theatre": "database_lecture_theatre",
    "office": "database_office",
}

# Published state-of-the-art mean Euclidean positioning error (meters) that
# this pipeline is trying to beat.
SOTA_MEAN_ERROR_M = {
    "corridor": 1.85,
    "lecture_theatre": 1.23,
    "office": 1.98,
}


@dataclasses.dataclass
class EnvironmentData:
    env_name: str
    X_train: np.ndarray
    y_train: np.ndarray  # shape (n, 2) -> [X, Y]
    X_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    active_aps: list[int]  # AP indices (1-based) kept for this environment


def _raw_columns(ap: int) -> tuple[str, str]:
    return f"AP{ap} RTT(mm)", f"AP{ap} RSS(dBm)"


def load_raw(env_name: str, data_dir: str | Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load the raw train/test CSVs for one environment."""
    if env_name not in ENVIRONMENTS:
        raise ValueError(f"Unknown environment '{env_name}'. Choose from {list(ENVIRONMENTS)}")
    data_dir = Path(data_dir)
    prefix = ENVIRONMENTS[env_name]
    train_df = pd.read_csv(data_dir / f"{prefix}_train.csv")
    test_df = pd.read_csv(data_dir / f"{prefix}_test.csv")
    return train_df, test_df


def build_features(env_name: str, data_dir: str | Path) -> EnvironmentData:
    """Load + preprocess one environment into model-ready arrays."""
    train_df, test_df = load_raw(env_name, data_dir)

    active_aps = []
    cap_high = {}   # ap -> value to replace sentinel RTT with
    cap_low = {}    # ap -> value to replace sentinel RSS with

    for ap in range(1, N_APS + 1):
        rtt_col, rss_col = _raw_columns(ap)
        detected_mask = train_df[rtt_col] != RTT_SENTINEL
        if detected_mask.sum() == 0:
            # Never detected in training data at all -> uninformative, drop.
            continue
        active_aps.append(ap)
        cap_high[ap] = train_df.loc[detected_mask, rtt_col].max() * 1.2
        cap_low[ap] = train_df.loc[detected_mask, rss_col].min() - 5.0

    def _transform(df: pd.DataFrame) -> pd.DataFrame:
        feats = {}
        for ap in active_aps:
            rtt_col, rss_col = _raw_columns(ap)
            sentinel_mask = df[rtt_col] == RTT_SENTINEL
            rtt = df[rtt_col].where(~sentinel_mask, cap_high[ap])
            rss = df[rss_col].where(~sentinel_mask, cap_low[ap])
            feats[f"AP{ap}_RTT"] = rtt
            feats[f"AP{ap}_RSS"] = rss
            feats[f"AP{ap}_detected"] = (~sentinel_mask).astype(float)
        return pd.DataFrame(feats)

    train_feats = _transform(train_df)
    test_feats = _transform(test_df)

    return EnvironmentData(
        env_name=env_name,
        X_train=train_feats.to_numpy(dtype=np.float64),
        y_train=train_df[["X", "Y"]].to_numpy(dtype=np.float64),
        X_test=test_feats.to_numpy(dtype=np.float64),
        y_test=test_df[["X", "Y"]].to_numpy(dtype=np.float64),
        feature_names=list(train_feats.columns),
        active_aps=active_aps,
    )


if __name__ == "__main__":
    # Quick sanity check when run directly.
    for env in ENVIRONMENTS:
        d = build_features(env, Path(__file__).parent / "data")
        print(
            f"{env}: train={d.X_train.shape} test={d.X_test.shape} "
            f"active_aps={d.active_aps} features={d.feature_names}"
        )
