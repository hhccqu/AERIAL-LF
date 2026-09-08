"""Dataset contracts and route batching for AERIAL-LF."""
from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .geometry import pnp_quality_gate


REQUIRED_FEATURE_KEYS = {
    "scene_id",
    "scene_query",
    "uav_tokens",
    "uav_token_valid_mask",
    "uav_token_anchor_ego_m",
    "route_scene_id",
    "route_command",
    "route_branch",
    "route_target_6x2",
    "route_pnp_rms_px",
    "route_group_id",
}

SEMANTIC_FIELDS = {
    "drivable": "reviewed_candidate_drivable",
    "obstacle": "reviewed_candidate_obstacle",
    "invalid": "invalid",
    "unknown": "unknown_reviewed",
    "grid": "grid_body_xy_m",
}


@dataclass
class FeatureDataset:
    arrays: dict[str, np.ndarray]
    semantic_maps: dict[str, dict[str, np.ndarray]]
    scene_row: np.ndarray

    @property
    def scene_ids(self) -> np.ndarray:
        return self.arrays["scene_id"].astype(str)

    @property
    def route_scene_ids(self) -> np.ndarray:
        return self.arrays["route_scene_id"].astype(str)

    @property
    def scene_count(self) -> int:
        return len(self.scene_ids)

    @property
    def route_count(self) -> int:
        return len(self.route_scene_ids)

    def route_indices(self, scene_ids: list[str]) -> np.ndarray:
        return np.flatnonzero(np.isin(self.route_scene_ids, np.asarray(scene_ids)))


def _require_shape(name: str, array: np.ndarray, tail: tuple[int, ...]) -> None:
    if array.ndim != len(tail) + 1 or tuple(array.shape[1:]) != tail:
        raise ValueError(f"{name} must have shape [N,{','.join(map(str, tail))}], got {array.shape}")


def validate_feature_arrays(arrays: dict[str, np.ndarray]) -> None:
    missing = REQUIRED_FEATURE_KEYS.difference(arrays)
    if missing:
        raise ValueError(f"feature contract missing keys: {sorted(missing)}")
    scene_count = len(arrays["scene_id"])
    route_count = len(arrays["route_scene_id"])
    if scene_count == 0 or route_count == 0:
        raise ValueError("feature archive must contain at least one scene and route")
    if len(set(arrays["scene_id"].astype(str))) != scene_count:
        raise ValueError("scene_id values must be unique")
    _require_shape("scene_query", arrays["scene_query"], (16, 256))
    _require_shape("uav_tokens", arrays["uav_tokens"], (36, 768))
    _require_shape("uav_token_valid_mask", arrays["uav_token_valid_mask"], (36,))
    _require_shape("uav_token_anchor_ego_m", arrays["uav_token_anchor_ego_m"], (36, 2))
    _require_shape("route_target_6x2", arrays["route_target_6x2"], (6, 2))
    for key in ("route_command", "route_branch", "route_pnp_rms_px", "route_group_id"):
        if len(arrays[key]) != route_count:
            raise ValueError(f"{key} route count does not match route_scene_id")
    if scene_count != len(arrays["scene_query"]):
        raise ValueError("scene-level array counts do not match scene_id")
    branches = np.asarray(arrays["route_branch"])
    if not np.isin(branches, (0, 1, 2)).all():
        raise ValueError("route_branch values must be 0, 1, or 2")
    finite_keys = (
        "scene_query",
        "uav_tokens",
        "uav_token_anchor_ego_m",
        "route_target_6x2",
        "route_pnp_rms_px",
    )
    for key in finite_keys:
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"{key} contains non-finite values")


def _semantic_path(record: dict[str, Any], registry: Path, roots: tuple[Path, ...]) -> Path:
    value = record["semantic_npz"]
    raw = Path(value["path"] if isinstance(value, dict) else value)
    candidates = [raw] if raw.is_absolute() else [registry.parent / raw]
    for root in roots:
        candidates.extend((
            root / raw,
            root / record["scene_id"] / raw.name,
            root / record["scene_id"] / "bev" / raw.name,
        ))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    attempted = "\n  ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"semantic map not found for {record['scene_id']}; tried:\n  {attempted}")


def load_semantic_maps(
    registry_path: str | Path,
    search_roots: tuple[Path, ...] = (),
) -> dict[str, dict[str, np.ndarray]]:
    registry = Path(registry_path).resolve()
    document = json.loads(registry.read_text(encoding="utf-8"))
    records = document.get("records")
    if not isinstance(records, list) or not records:
        raise ValueError("semantic registry must contain a non-empty records list")
    maps: dict[str, dict[str, np.ndarray]] = {}
    for record in records:
        scene_id = str(record["scene_id"])
        if scene_id in maps:
            raise ValueError(f"duplicate semantic scene: {scene_id}")
        path = _semantic_path(record, registry, search_roots)
        with np.load(path, allow_pickle=False) as source:
            missing = set(SEMANTIC_FIELDS.values()).difference(source.files)
            if missing:
                raise ValueError(f"{path} missing semantic fields: {sorted(missing)}")
            semantic = np.stack((
                source[SEMANTIC_FIELDS["drivable"]],
                source[SEMANTIC_FIELDS["obstacle"]],
            )).astype(np.float32)
            invalid = source[SEMANTIC_FIELDS["invalid"]].astype(bool)
            unknown = source[SEMANTIC_FIELDS["unknown"]].astype(bool)
            grid = source[SEMANTIC_FIELDS["grid"]].astype(np.float32)
        if semantic.ndim != 3 or semantic.shape[0] != 2:
            raise ValueError(f"{scene_id}: semantic map must be [2,H,W]")
        height, width = semantic.shape[1:]
        if invalid.shape != (height, width) or unknown.shape != (height, width):
            raise ValueError(f"{scene_id}: semantic masks do not match semantic map")
        if grid.shape != (height, width, 2):
            raise ValueError(f"{scene_id}: grid must be [H,W,2]")
        if not np.isfinite(semantic).all() or not np.isfinite(grid).all():
            raise ValueError(f"{scene_id}: semantic asset contains non-finite values")
        maps[scene_id] = {
            "semantic": semantic,
            "invalid": invalid,
            "unknown": unknown,
            "grid": grid,
            "source_path": np.asarray(str(path)),
        }
    return maps


def load_dataset(
    feature_path: str | Path,
    semantic_registry: str | Path,
    semantic_search_roots: tuple[Path, ...] = (),
) -> FeatureDataset:
    arrays, scene_row = load_feature_archive(feature_path)
    maps = load_semantic_maps(semantic_registry, semantic_search_roots)
    expected = set(arrays["scene_id"].astype(str))
    if set(maps) != expected:
        raise ValueError(
            "semantic scene coverage mismatch: "
            f"missing={sorted(expected.difference(maps))}, extra={sorted(set(maps).difference(expected))}"
        )
    return FeatureDataset(arrays, maps, scene_row)


def load_feature_archive(feature_path: str | Path) -> tuple[dict[str, np.ndarray], np.ndarray]:
    with np.load(feature_path, allow_pickle=False) as source:
        arrays = {key: np.array(source[key], copy=True) for key in source.files}
    validate_feature_arrays(arrays)
    scene_rows = {
        scene_id: index for index, scene_id in enumerate(arrays["scene_id"].astype(str))
    }
    missing_routes = sorted(set(arrays["route_scene_id"].astype(str)).difference(scene_rows))
    if missing_routes:
        raise ValueError(f"routes reference unknown scenes: {missing_routes}")
    scene_row = np.asarray([
        scene_rows[value] for value in arrays["route_scene_id"].astype(str)
    ])
    return arrays, scene_row


def validate_split(dataset: FeatureDataset, split: dict[str, Any]) -> None:
    folds = split.get("folds")
    if not isinstance(folds, list) or not folds:
        raise ValueError("split must contain a non-empty folds list")
    all_scenes = set(dataset.scene_ids)
    held_out: list[str] = []
    for fold_index, fold in enumerate(folds):
        partitions = [set(fold[f"{name}_scenes"]) for name in ("train", "val", "test")]
        if any(partitions[i] & partitions[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"fold {fold_index} has overlapping scene partitions")
        if set.union(*partitions) != all_scenes:
            raise ValueError(f"fold {fold_index} does not partition all scenes")
        route_parts = [dataset.route_indices(fold[f"{name}_scenes"]) for name in ("train", "val", "test")]
        group_parts = [set(dataset.arrays["route_group_id"][indices].astype(str)) for indices in route_parts]
        if any(group_parts[i] & group_parts[j] for i, j in ((0, 1), (0, 2), (1, 2))):
            raise ValueError(f"fold {fold_index} has overlapping route groups")
        held_out.extend(fold["test_scenes"])
    if sorted(held_out) != sorted(dataset.scene_ids.tolist()):
        raise ValueError("each scene must appear in exactly one held-out fold")


def load_split(path: str | Path, dataset: FeatureDataset) -> dict[str, Any]:
    split = json.loads(Path(path).read_text(encoding="utf-8"))
    validate_split(dataset, split)
    return split


def make_batch(dataset: FeatureDataset, indices: np.ndarray, device: torch.device):
    indices = np.asarray(indices)
    scene_rows = dataset.scene_row[indices]
    maps = [dataset.semantic_maps[scene] for scene in dataset.route_scene_ids[indices]]
    gate = pnp_quality_gate(dataset.arrays["route_pnp_rms_px"][indices])
    return (
        torch.as_tensor(dataset.arrays["scene_query"][scene_rows], device=device),
        torch.as_tensor(dataset.arrays["uav_tokens"][scene_rows], device=device),
        torch.as_tensor(dataset.arrays["uav_token_valid_mask"][scene_rows], device=device),
        torch.as_tensor(dataset.arrays["uav_token_anchor_ego_m"][scene_rows], device=device),
        torch.as_tensor(np.stack([item["semantic"] for item in maps]), device=device),
        torch.as_tensor(np.stack([item["grid"] for item in maps]), device=device),
        torch.as_tensor(np.stack([item["invalid"] for item in maps]), device=device),
        torch.as_tensor(np.stack([item["unknown"] for item in maps]), device=device),
        torch.as_tensor(gate, device=device),
        torch.as_tensor(dataset.arrays["route_target_6x2"][indices], device=device),
    )


def selected_branch_loss(prediction, target, branch):
    rows = torch.arange(len(branch), device=prediction.device)
    return torch.abs(prediction[rows, branch] - target).mean()
