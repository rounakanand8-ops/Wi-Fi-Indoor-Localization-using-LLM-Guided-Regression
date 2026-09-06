"""
Serialize a preprocessed feature row (see data_utils.py) into a natural
language description the LLM backbone can tokenize.

Kept deliberately simple and consistent — one clause per active AP — so the
model sees a uniform template and the only thing that varies row-to-row is
the numbers themselves.
"""

from __future__ import annotations

import numpy as np


def serialize_row(feature_row: np.ndarray, feature_names: list[str]) -> str:
    """
    feature_names come in groups of 3 per AP: AP{i}_RTT, AP{i}_RSS, AP{i}_detected
    (see data_utils.build_features). Render each AP as one clause.
    """
    parts = []
    i = 0
    while i < len(feature_names):
        name = feature_names[i]
        ap_label = name.split("_")[0]  # e.g. "AP2"
        rtt = feature_row[i]
        rss = feature_row[i + 1]
        detected = feature_row[i + 2]
        status = "detected" if detected > 0.5 else "not detected (using fallback value)"
        parts.append(
            f"{ap_label}: round-trip time {rtt:.0f} millimeters, "
            f"signal strength {rss:.0f} dBm, {status}"
        )
        i += 3
    return "Wi-Fi access point readings — " + "; ".join(parts) + "."


def serialize_batch(feature_matrix: np.ndarray, feature_names: list[str]) -> list[str]:
    return [serialize_row(row, feature_names) for row in feature_matrix]
