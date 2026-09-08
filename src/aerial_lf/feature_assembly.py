"""Build a training feature archive without changing feature values."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .data import validate_feature_arrays
from .provenance import sha256


DINO_FIELDS = (
    "uav_tokens",
    "uav_token_anchor_ego_m",
    "uav_token_valid_mask",
)
SCENE_FIELDS = (
    "scene_id",
    "scene_query",
    "latent_pos",
    "uav_tokens",
    "uav_token_valid_mask",
    "uav_token_anchor_ego_m",
)
ROUTE_FIELDS = (
    "route_scene_id",
    "route_command",
    "route_branch",
    "route_target_6x2",
    "route_pnp_rms_px",
    "route_group_id",
)


def _load(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as source:
        return {key: np.array(source[key], copy=True) for key in source.files}


def replace_dino(base: Path, registry_path: Path, output: Path) -> dict:
    arrays = _load(base)
    validate_feature_arrays(arrays)
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    rows = {scene: index for index, scene in enumerate(arrays["scene_id"].astype(str))}
    replaced = []
    replacement_hashes = {}
    for record in registry["records"]:
        scene_id = str(record["scene_id"])
        if scene_id not in rows:
            raise ValueError(f"replacement references unknown scene: {scene_id}")
        path = Path(record["path"])
        if not path.is_absolute():
            path = (registry_path.parent / path).resolve()
        with np.load(path, allow_pickle=False) as source:
            token = np.asarray(source["uav_tokens_768"], dtype=np.float32)
            anchor = np.asarray(source["uav_token_anchor_ego_m"], dtype=np.float32)
            mask = np.asarray(source["uav_token_valid_mask"], dtype=bool)
        token = token[0] if token.shape == (1, 36, 768) else token
        anchor = anchor[0] if anchor.shape == (1, 36, 2) else anchor
        mask = mask[0] if mask.shape == (1, 36) else mask
        if token.shape != (36, 768) or anchor.shape != (36, 2) or mask.shape != (36,):
            raise ValueError(f"invalid DINO replacement shapes for {scene_id}")
        index = rows[scene_id]
        arrays["uav_tokens"][index] = token
        arrays["uav_token_anchor_ego_m"][index] = anchor
        arrays["uav_token_valid_mask"][index] = mask
        replaced.append(scene_id)
        replacement_hashes[scene_id] = sha256(path)
    validate_feature_arrays(arrays)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **arrays)
    return {
        "schema": "AERIAL_LF_DINO_REPLACEMENT_V01",
        "status": "pass",
        "source_sha256": sha256(base),
        "registry_sha256": sha256(registry_path),
        "output_sha256": sha256(output),
        "replaced_scenes": replaced,
        "replacement_hashes": replacement_hashes,
        "preserved_non_dino_fields": [key for key in arrays if key not in DINO_FIELDS],
    }


def merge_archives(sources: list[Path], output: Path) -> dict:
    if len(sources) < 2:
        raise ValueError("merge requires at least two source archives")
    archives = [_load(path) for path in sources]
    for arrays in archives:
        validate_feature_arrays(arrays)
    key_set = set(archives[0])
    if any(set(arrays) != key_set for arrays in archives[1:]):
        raise ValueError("all archives must have identical keys")
    scene_fields = []
    route_fields = []
    for key in key_set:
        is_scene = all(len(arrays[key]) == len(arrays["scene_id"]) for arrays in archives)
        is_route = all(len(arrays[key]) == len(arrays["route_scene_id"]) for arrays in archives)
        if key in ROUTE_FIELDS:
            route_fields.append(key)
        elif key in SCENE_FIELDS or (is_scene and not is_route):
            scene_fields.append(key)
        elif is_route and not is_scene:
            route_fields.append(key)
        else:
            raise ValueError(f"cannot infer scene/route ownership for field {key}")
    merged = {
        key: np.concatenate([arrays[key] for arrays in archives], axis=0)
        for key in scene_fields + route_fields
    }
    validate_feature_arrays(merged)
    scene_ids = merged["scene_id"].astype(str)
    if len(set(scene_ids)) != len(scene_ids):
        raise ValueError("merged archives contain duplicate scene IDs")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, **merged)
    return {
        "schema": "AERIAL_LF_FEATURE_MERGE_V01",
        "status": "pass",
        "sources": [
            {"path": str(path), "sha256": sha256(path)} for path in sources
        ],
        "output_sha256": sha256(output),
        "scene_count": int(len(merged["scene_id"])),
        "route_count": int(len(merged["route_scene_id"])),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    replace_parser = subparsers.add_parser("replace-dino")
    replace_parser.add_argument("--base", required=True)
    replace_parser.add_argument("--registry", required=True)
    replace_parser.add_argument("--output", required=True)
    merge_parser = subparsers.add_parser("merge")
    merge_parser.add_argument("--source", action="append", required=True)
    merge_parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.command == "replace-dino":
        result = replace_dino(Path(args.base).resolve(), Path(args.registry).resolve(), output)
    else:
        result = merge_archives([Path(path).resolve() for path in args.source], output)
    audit_path = output.with_suffix(".audit.json")
    audit_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
