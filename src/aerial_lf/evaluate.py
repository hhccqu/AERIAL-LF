"""Metrics and out-of-fold prediction aggregation."""
from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np


def waypoint_metrics(prediction: np.ndarray, target: np.ndarray, branch: np.ndarray) -> dict:
    branch = np.asarray(branch, dtype=np.int64)
    chosen = prediction[np.arange(len(branch)), branch]
    error = np.linalg.norm(chosen - target, axis=-1)
    return {
        "count": int(len(error)),
        "ade_m": float(error.mean()),
        "fde_090_m": float(error[:, -1].mean()),
    }


def aggregate_prediction_files(
    paths: Iterable[str | Path],
    expected_routes: int,
    expected_predictions_per_route: int,
) -> tuple[dict, dict[str, np.ndarray]]:
    paths = [Path(path) for path in paths]
    grouped: dict[int, list[tuple[np.ndarray, np.ndarray, int]]] = {}
    for path in paths:
        with np.load(path, allow_pickle=False) as arrays:
            for row, route_index in enumerate(arrays["route_index"]):
                grouped.setdefault(int(route_index), []).append((
                    np.array(arrays["prediction"][row], copy=True),
                    np.array(arrays["target"][row], copy=True),
                    int(arrays["branch"][row]),
                ))
    if set(grouped) != set(range(expected_routes)):
        raise RuntimeError("OOF files do not cover every route exactly by route_index")
    if not all(len(rows) == expected_predictions_per_route for rows in grouped.values()):
        raise RuntimeError("unexpected number of held-out predictions per route")
    ordered = [grouped[index] for index in range(expected_routes)]
    for route_index, rows in enumerate(ordered):
        if not all(np.array_equal(rows[0][1], item[1]) and rows[0][2] == item[2] for item in rows[1:]):
            raise RuntimeError(f"target or branch mismatch for route {route_index}")
    prediction = np.stack([
        np.mean(np.stack([item[0] for item in rows]), axis=0) for rows in ordered
    ])
    target = np.stack([rows[0][1] for rows in ordered])
    branch = np.asarray([rows[0][2] for rows in ordered])
    chosen = prediction[np.arange(len(branch)), branch]
    arrays = {"prediction": prediction, "target": target, "branch": branch}
    result = {
        "schema": "AERIAL_LF_OOF_AGGREGATE_V01",
        "status": "completed",
        "route_count": expected_routes,
        "run_count": len(paths),
        "metrics": {**waypoint_metrics(prediction, target, branch), "seed_averaged": True},
        "checks": {
            "all_finite": bool(np.isfinite(prediction).all()),
            "all_forward_monotone": bool((np.diff(chosen[..., 0], axis=-1) >= -1e-7).all()),
        },
    }
    return result, arrays
