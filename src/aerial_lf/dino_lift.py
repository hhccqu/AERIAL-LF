"""Lift frozen DINO patch features onto the fixed body-frame grid."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .provenance import sha256


def project(points_world: np.ndarray, homography: np.ndarray) -> np.ndarray:
    flat = points_world.reshape(-1, 2)
    homogeneous = np.c_[flat, np.ones(len(flat), dtype=np.float64)] @ homography.T
    if np.any(~np.isfinite(homogeneous)) or np.any(np.abs(homogeneous[:, 2]) < 1e-12):
        raise ValueError("non-finite or infinite homography projection")
    return (homogeneous[:, :2] / homogeneous[:, 2:3]).reshape(points_world.shape)


def bilinear(feature_map: np.ndarray, coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    height, width, _ = feature_map.shape
    xy = np.asarray(coordinates, dtype=np.float32).reshape(-1, 2)
    valid = np.isfinite(xy).all(axis=1)
    valid &= (
        (xy[:, 0] >= 0)
        & (xy[:, 0] <= width - 1)
        & (xy[:, 1] >= 0)
        & (xy[:, 1] <= height - 1)
    )
    x = np.clip(xy[:, 0], 0, width - 1)
    y = np.clip(xy[:, 1], 0, height - 1)
    x0, y0 = np.floor(x).astype(np.int64), np.floor(y).astype(np.int64)
    x1, y1 = np.minimum(x0 + 1, width - 1), np.minimum(y0 + 1, height - 1)
    wx, wy = (x - x0)[:, None], (y - y0)[:, None]
    top = feature_map[y0, x0] * (1 - wx) + feature_map[y0, x1] * wx
    bottom = feature_map[y1, x0] * (1 - wx) + feature_map[y1, x1] * wx
    values = top * (1 - wy) + bottom * wy
    values[~valid] = 0
    return values.astype(np.float32), valid


def position_encoding(anchors: np.ndarray, dim: int = 768) -> np.ndarray:
    forward = np.clip(2.0 * anchors[:, 0] / 0.9 - 1.0, -1.0, 1.0)
    left = np.clip(2.0 * (anchors[:, 1] + 0.9) / 1.8 - 1.0, -1.0, 1.0)
    frequency = np.arange(dim // 4, dtype=np.float32) + 1.0
    return 0.01 * np.concatenate((
        np.sin(np.pi * forward[:, None] * frequency),
        np.cos(np.pi * forward[:, None] * frequency),
        np.sin(np.pi * left[:, None] * frequency),
        np.cos(np.pi * left[:, None] * frequency),
    ), axis=1)[:, :dim].astype(np.float32)


def lift_scene(
    patch: np.ndarray,
    preprocess: dict,
    geometry: dict,
    grid: np.ndarray,
) -> dict[str, np.ndarray]:
    if patch.shape != (1, 37, 37, 768):
        raise ValueError(f"patch_features must be [1,37,37,768], got {patch.shape}")
    if grid.shape != (180, 360, 2):
        raise ValueError(f"grid_body_xy_m must be [180,360,2], got {grid.shape}")
    homography = np.asarray(geometry["homography_W_tag_to_image"], dtype=np.float64)
    pose = geometry["vehicle_pose"]
    center = np.asarray(pose["body_reference_W_tag_m"][:2], dtype=np.float64)
    theta = np.deg2rad(float(pose["yaw_body_deg"]))
    rotation = np.array(((np.cos(theta), -np.sin(theta)), (np.sin(theta), np.cos(theta))))
    world = grid.astype(np.float64) @ rotation.T + center
    pixels = project(world, homography)
    original_h, original_w = [int(value) for value in preprocess["original_hw"]]
    resized_h, resized_w = [int(value) for value in preprocess["resized_hw"]]
    pad_top, pad_left = [int(value) for value in preprocess["pad_top_left"]]
    feature_xy = np.stack((
        (pixels[..., 0] * resized_w / original_w + pad_left) / 14.0 - 0.5,
        (pixels[..., 1] * resized_h / original_h + pad_top) / 14.0 - 0.5,
    ), axis=-1).astype(np.float32)
    image_valid = (
        (pixels[..., 0] >= 0)
        & (pixels[..., 0] < original_w)
        & (pixels[..., 1] >= 0)
        & (pixels[..., 1] < original_h)
    )
    valid = np.zeros((180, 360), dtype=bool)
    tokens = np.zeros((36, 768), dtype=np.float32)
    anchors = np.zeros((36, 2), dtype=np.float32)
    fractions = np.zeros(36, dtype=np.float32)
    token_mask = np.zeros(36, dtype=bool)
    for row in range(6):
        r0, r1 = row * 30, (row + 1) * 30
        sampled, plane_valid = bilinear(patch[0], feature_xy[r0:r1])
        sampled = sampled.reshape(30, 360, 768)
        band_valid = plane_valid.reshape(30, 360) & image_valid[r0:r1]
        valid[r0:r1] = band_valid
        sampled[~band_valid] = 0
        for col in range(6):
            index = row * 6 + col
            c0, c1 = col * 60, (col + 1) * 60
            region_valid = band_valid[:, c0:c1]
            fractions[index] = region_valid.mean()
            if region_valid.any():
                tokens[index] = sampled[:, c0:c1][region_valid].mean(axis=0)
                anchors[index] = grid[r0:r1, c0:c1][region_valid].mean(axis=0)
                token_mask[index] = True
    tokens += position_encoding(anchors)
    tokens[~token_mask] = 0
    return {
        "dense_uav_bev_valid_mask": valid,
        "grid_body_xy_m": grid,
        "uav_image_pixels": pixels.astype(np.float32),
        "dino_feature_xy": feature_xy,
        "uav_tokens_768": tokens[None],
        "uav_token_anchor_ego_m": anchors[None],
        "uav_token_valid_fraction": fractions[None],
        "uav_token_valid_mask": token_mask[None],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve()
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir).resolve()
    records = []
    for record in registry["records"]:
        scene_id = str(record["scene_id"])
        resolved = {}
        for key in ("patch_npz", "patch_manifest", "geometry", "grid_npz"):
            path = Path(record[key])
            resolved[key] = path if path.is_absolute() else (registry_path.parent / path).resolve()
        with np.load(resolved["patch_npz"], allow_pickle=False) as source:
            patch = np.asarray(source["patch_features"], dtype=np.float32)
        manifest = json.loads(resolved["patch_manifest"].read_text(encoding="utf-8"))
        geometry = json.loads(resolved["geometry"].read_text(encoding="utf-8"))
        with np.load(resolved["grid_npz"], allow_pickle=False) as source:
            grid = np.asarray(source["grid_body_xy_m"], dtype=np.float32)
        payload = lift_scene(patch, manifest["encoder"]["preprocess"], geometry, grid)
        scene_dir = output_dir / scene_id
        scene_dir.mkdir(parents=True, exist_ok=True)
        output = scene_dir / f"{scene_id}_canonical_dino_tokens.npz"
        np.savez_compressed(output, **payload)
        records.append({
            "scene_id": scene_id,
            "path": str(output),
            "sha256": sha256(output),
            "valid_tokens": int(payload["uav_token_valid_mask"].sum()),
            "pnp_rms_px": geometry.get("field_reprojection_rms_px"),
        })
    report = {
        "schema": "AERIAL_LF_DINO_LIFT_V01",
        "status": "completed",
        "registry_sha256": sha256(registry_path),
        "records": records,
    }
    (output_dir / "dino_lift_registry.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

