import json

import numpy as np
import pytest
import torch

from aerial_lf.baseline import VehicleOnlyModel
from aerial_lf.config import BaselineConfig, ExperimentConfig, PathsConfig, TrainingConfig
from aerial_lf.data import load_dataset, load_split
from aerial_lf.evaluate import aggregate_prediction_files
from aerial_lf.feature_assembly import merge_archives
from aerial_lf.train import train_one


def make_dataset(tmp_path):
    scene_ids = np.asarray(["scene_0", "scene_1", "scene_2"])
    feature_path = tmp_path / "features.npz"
    np.savez_compressed(
        feature_path,
        scene_id=scene_ids,
        scene_query=np.zeros((3, 16, 256), np.float32),
        uav_tokens=np.zeros((3, 36, 768), np.float32),
        uav_token_valid_mask=np.ones((3, 36), bool),
        uav_token_anchor_ego_m=np.zeros((3, 36, 2), np.float32),
        route_scene_id=scene_ids,
        route_command=np.asarray(["right", "left", "straight"]),
        route_branch=np.asarray([0, 1, 2], np.int64),
        route_target_6x2=np.zeros((3, 6, 2), np.float32),
        route_pnp_rms_px=np.asarray([2, 6, 9], np.float32),
        route_group_id=np.asarray(["group_0", "group_1", "group_2"]),
    )
    records = []
    for scene_id in scene_ids:
        path = tmp_path / f"{scene_id}.npz"
        np.savez_compressed(
            path,
            reviewed_candidate_drivable=np.ones((8, 8), np.float32),
            reviewed_candidate_obstacle=np.zeros((8, 8), np.float32),
            invalid=np.zeros((8, 8), bool),
            unknown_reviewed=np.zeros((8, 8), bool),
            grid_body_xy_m=np.zeros((8, 8, 2), np.float32),
        )
        records.append({"scene_id": str(scene_id), "semantic_npz": {"path": path.name}})
    registry_path = tmp_path / "registry.json"
    registry_path.write_text(json.dumps({"records": records}), encoding="utf-8")
    folds = []
    for test in range(3):
        val = (test + 1) % 3
        train = (test + 2) % 3
        folds.append({
            "fold": test,
            "train_scenes": [str(scene_ids[train])],
            "val_scenes": [str(scene_ids[val])],
            "test_scenes": [str(scene_ids[test])],
        })
    split_path = tmp_path / "split.json"
    split_path.write_text(json.dumps({"folds": folds}), encoding="utf-8")
    return feature_path, registry_path, split_path


def test_dataset_and_split_contract(tmp_path):
    feature_path, registry_path, split_path = make_dataset(tmp_path)
    dataset = load_dataset(feature_path, registry_path)
    split = load_split(split_path, dataset)
    assert dataset.scene_count == 3
    assert dataset.route_count == 3
    assert len(split["folds"]) == 3


def test_group_leakage_is_rejected(tmp_path):
    feature_path, registry_path, split_path = make_dataset(tmp_path)
    with np.load(feature_path, allow_pickle=False) as source:
        arrays = {key: source[key] for key in source.files}
    arrays["route_group_id"] = np.asarray(["shared", "shared", "group_2"])
    np.savez_compressed(feature_path, **arrays)
    dataset = load_dataset(feature_path, registry_path)
    with pytest.raises(ValueError, match="route groups"):
        load_split(split_path, dataset)


def test_routewise_prediction_ensemble(tmp_path):
    target = np.zeros((3, 6, 2), np.float32)
    branch = np.asarray([0, 1, 2])
    paths = []
    for seed, offset in enumerate((1.0, 3.0)):
        prediction = np.zeros((3, 3, 6, 2), np.float32)
        prediction[:, :, :, 0] = offset
        path = tmp_path / f"seed_{seed}.npz"
        np.savez_compressed(
            path,
            prediction=prediction,
            target=target,
            branch=branch,
            route_index=np.arange(3),
        )
        paths.append(path)
    result, arrays = aggregate_prediction_files(paths, 3, 2)
    assert np.all(arrays["prediction"][..., 0] == 2.0)
    assert result["metrics"]["ade_m"] == pytest.approx(2.0)
    assert result["run_count"] == 2


def test_one_epoch_training_smoke(tmp_path):
    feature_path, registry_path, split_path = make_dataset(tmp_path)
    checkpoint_pattern = str(tmp_path / "baseline" / "fold{fold}_seed{seed}" / "M0.pth")
    checkpoint = tmp_path / "baseline" / "fold0_seed11" / "M0.pth"
    checkpoint.parent.mkdir(parents=True)
    torch.save(VehicleOnlyModel().state_dict(), checkpoint)
    source = tmp_path / "config.toml"
    source.write_text("# synthetic smoke config\n", encoding="utf-8")
    config = ExperimentConfig(
        source=source,
        paths=PathsConfig(
            features=feature_path,
            split=split_path,
            semantic_registry=registry_path,
            semantic_search_roots=(),
            baseline_checkpoint=checkpoint_pattern,
            output_dir=tmp_path / "outputs",
        ),
        training=TrainingConfig(
            seeds=(11,), epochs=1, batch_size=1, num_threads=1, device="cpu"
        ),
        baseline=BaselineConfig(),
    )
    dataset = load_dataset(feature_path, registry_path)
    split = load_split(split_path, dataset)
    result = train_one(dataset, split, config, 0, 11, torch.device("cpu"), epochs=1)
    assert result["status"] == "completed"
    assert result["scope"] == {"train_routes": 1, "val_routes": 1, "test_routes": 1}
    assert (config.paths.output_dir / "fold0_seed11" / "predictions.npz").exists()


def test_merge_supports_one_route_per_scene(tmp_path):
    sources = []
    for index in range(2):
        path = tmp_path / f"part_{index}.npz"
        scene_id = np.asarray([f"scene_{index}"])
        np.savez_compressed(
            path,
            scene_id=scene_id,
            scene_query=np.zeros((1, 16, 256), np.float32),
            uav_tokens=np.zeros((1, 36, 768), np.float32),
            uav_token_valid_mask=np.ones((1, 36), bool),
            uav_token_anchor_ego_m=np.zeros((1, 36, 2), np.float32),
            route_scene_id=scene_id,
            route_command=np.asarray(["right"]),
            route_branch=np.asarray([0]),
            route_target_6x2=np.zeros((1, 6, 2), np.float32),
            route_pnp_rms_px=np.asarray([2], np.float32),
            route_group_id=np.asarray([f"group_{index}"]),
        )
        sources.append(path)
    output = tmp_path / "merged.npz"
    result = merge_archives(sources, output)
    assert result["scene_count"] == 2
    assert result["route_count"] == 2
