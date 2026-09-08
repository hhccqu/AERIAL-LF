"""Read-only audit of configured features, semantics, split, and checkpoints."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .config import load_config
from .data import load_dataset, load_split
from .model import F15JointHeadModel, trainable_parameter_count
from .provenance import sha256


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output")
    parser.add_argument("--require-checkpoints", action="store_true")
    args = parser.parse_args()
    config = load_config(args.config)
    dataset = load_dataset(
        config.paths.features,
        config.paths.semantic_registry,
        config.paths.semantic_search_roots,
    )
    split = load_split(config.paths.split, dataset)
    checkpoint_rows = []
    for fold in range(len(split["folds"])):
        for seed in config.training.seeds:
            path = config.paths.checkpoint(fold, seed)
            if args.require_checkpoints and not path.exists():
                raise FileNotFoundError(path)
            checkpoint_rows.append({
                "fold": fold,
                "seed": seed,
                "present": path.exists(),
                "sha256": sha256(path) if path.exists() else None,
            })
    branches, counts = np.unique(dataset.arrays["route_branch"], return_counts=True)
    report = {
        "schema": "AERIAL_LF_INPUT_AUDIT_V01",
        "status": "pass",
        "scope": {
            "scenes": dataset.scene_count,
            "routes": dataset.route_count,
            "groups": int(len(set(dataset.arrays["route_group_id"].astype(str)))),
            "folds": len(split["folds"]),
            "seeds": list(config.training.seeds),
            "branch_counts": {str(int(key)): int(value) for key, value in zip(branches, counts)},
        },
        "feature_shapes": {
            key: list(dataset.arrays[key].shape)
            for key in (
                "scene_query",
                "uav_tokens",
                "uav_token_valid_mask",
                "uav_token_anchor_ego_m",
                "route_target_6x2",
            )
        },
        "semantic_shapes": {
            "semantic": list(next(iter(dataset.semantic_maps.values()))["semantic"].shape),
            "grid": list(next(iter(dataset.semantic_maps.values()))["grid"].shape),
        },
        "trainable_parameters": trainable_parameter_count(F15JointHeadModel()),
        "hashes": {
            "features": sha256(config.paths.features),
            "split": sha256(config.paths.split),
            "semantic_registry": sha256(config.paths.semantic_registry),
            "config": sha256(config.source),
        },
        "baseline_checkpoints": checkpoint_rows,
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
    print(text, end="")


if __name__ == "__main__":
    main()

