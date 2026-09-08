"""Five-fold, multi-seed AERIAL-LF training and OOF evaluation."""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch

from .config import ExperimentConfig, load_config
from .data import FeatureDataset, load_dataset, load_split, make_batch, selected_branch_loss
from .evaluate import aggregate_prediction_files, waypoint_metrics
from .model import F15JointHeadModel, trainable_parameter_count
from .provenance import sha256


def seed_all(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.use_deterministic_algorithms(True, warn_only=True)


def select_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def _load_checkpoint(path: Path, device: torch.device):
    try:
        return torch.load(path, map_location=device, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=device)


def train_one(
    dataset: FeatureDataset,
    split: dict,
    config: ExperimentConfig,
    fold: int,
    seed: int,
    device: torch.device,
    epochs: int | None = None,
) -> dict:
    training = config.training
    epochs = training.epochs if epochs is None else epochs
    seed_all(seed)
    fold_spec = split["folds"][fold]
    train_indices = dataset.route_indices(fold_spec["train_scenes"])
    val_indices = dataset.route_indices(fold_spec["val_scenes"])
    test_indices = dataset.route_indices(fold_spec["test_scenes"])
    output = config.paths.output_dir / f"fold{fold}_seed{seed}"
    output.mkdir(parents=True, exist_ok=True)

    model = F15JointHeadModel().to(device)
    baseline_path = config.paths.checkpoint(fold, seed)
    loaded = model.load_state_dict(_load_checkpoint(baseline_path, device), strict=False)
    invalid_missing = [
        key for key in loaded.missing_keys
        if not key.startswith(("dino.", "dense_semantic."))
    ]
    if loaded.unexpected_keys or invalid_missing:
        raise RuntimeError(f"baseline checkpoint contract failed: {loaded}")

    optimizer = torch.optim.AdamW(
        [
            {
                "params": list(model.dino.parameters())
                + list(model.dense_semantic.parameters()),
                "lr": training.fusion_lr,
            },
            {"params": list(model.head.parameters()), "lr": training.head_lr},
        ],
        weight_decay=training.weight_decay,
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    for epoch in range(1, epochs + 1):
        model.train()
        order = np.random.default_rng(seed + 1000 * fold + epoch).permutation(train_indices)
        train_losses = []
        for start in range(0, len(order), training.batch_size):
            indices = order[start:start + training.batch_size]
            values = make_batch(dataset, indices, device)
            prediction, _ = model(*values[:-1])
            branch = torch.as_tensor(
                dataset.arrays["route_branch"][indices], device=device
            )
            current = selected_branch_loss(prediction, values[-1], branch)
            optimizer.zero_grad(set_to_none=True)
            current.backward()
            optimizer.step()
            train_losses.append(float(current.detach()))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for start in range(0, len(val_indices), training.batch_size):
                indices = val_indices[start:start + training.batch_size]
                values = make_batch(dataset, indices, device)
                prediction, _ = model(*values[:-1])
                branch = torch.as_tensor(
                    dataset.arrays["route_branch"][indices], device=device
                )
                val_losses.append(float(selected_branch_loss(prediction, values[-1], branch)))
        val_loss = float(np.mean(val_losses))
        history.append({
            "epoch": epoch,
            "train_l1_m": float(np.mean(train_losses)),
            "val_l1_m": val_loss,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch

    if best_state is None:
        raise RuntimeError("training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    prediction_parts = []
    with torch.no_grad():
        for start in range(0, len(test_indices), training.batch_size):
            indices = test_indices[start:start + training.batch_size]
            values = make_batch(dataset, indices, device)
            prediction_parts.append(model(*values[:-1])[0].cpu().numpy())
    prediction = np.concatenate(prediction_parts)
    target = dataset.arrays["route_target_6x2"][test_indices]
    branch = dataset.arrays["route_branch"][test_indices]

    np.savez_compressed(
        output / "predictions.npz",
        prediction=prediction,
        target=target,
        branch=branch,
        scene_id=dataset.route_scene_ids[test_indices],
        route_index=test_indices,
        route_group_id=dataset.arrays["route_group_id"][test_indices],
        route_command=dataset.arrays["route_command"][test_indices],
        pnp=dataset.arrays["route_pnp_rms_px"][test_indices],
    )
    torch.save(model.state_dict(), output / "AERIAL_LF.pth")
    (output / "history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    result = {
        "schema": "AERIAL_LF_RUN_V01",
        "status": "completed",
        "fold": fold,
        "seed": seed,
        "device": str(device),
        "scope": {
            "train_routes": len(train_indices),
            "val_routes": len(val_indices),
            "test_routes": len(test_indices),
        },
        "best_epoch": best_epoch,
        "best_val_l1_m": best_loss,
        "metrics": waypoint_metrics(prediction, target, branch),
        "training": {
            "epochs": epochs,
            "batch_size": training.batch_size,
            "fusion_lr": training.fusion_lr,
            "head_lr": training.head_lr,
            "weight_decay": training.weight_decay,
            "head_initialization": "fold- and seed-specific vehicle-only baseline",
            "head_training": "fine-tuned",
            "validation_only_selection": True,
            "test_evaluated_once": True,
            "trainable_parameters": trainable_parameter_count(model),
        },
        "inputs": {
            "feature_path": str(config.paths.features),
            "feature_sha256": sha256(config.paths.features),
            "split_sha256": sha256(config.paths.split),
            "semantic_registry_sha256": sha256(config.paths.semantic_registry),
            "baseline_checkpoint_sha256": sha256(baseline_path),
            "config_sha256": sha256(config.source),
        },
    }
    (output / "run_result.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def aggregate_suite(config: ExperimentConfig, dataset: FeatureDataset, split: dict) -> dict:
    paths = [
        config.paths.output_dir / f"fold{fold}_seed{seed}" / "predictions.npz"
        for fold in range(len(split["folds"]))
        for seed in config.training.seeds
    ]
    result, arrays = aggregate_prediction_files(
        paths,
        expected_routes=dataset.route_count,
        expected_predictions_per_route=len(config.training.seeds),
    )
    result["protocol"] = {
        "folds": len(split["folds"]),
        "seeds": list(config.training.seeds),
        "fusion_lr": config.training.fusion_lr,
        "head_lr": config.training.head_lr,
    }
    np.savez_compressed(config.paths.output_dir / "aggregate_predictions.npz", **arrays)
    (config.paths.output_dir / "aggregate_results.json").write_text(
        json.dumps(result, indent=2) + "\n", encoding="utf-8"
    )
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config.training.num_threads)
    device = select_device(config.training.device)
    dataset = load_dataset(
        config.paths.features,
        config.paths.semantic_registry,
        config.paths.semantic_search_roots,
    )
    split = load_split(config.paths.split, dataset)
    config.paths.output_dir.mkdir(parents=True, exist_ok=True)
    if args.fold not in range(len(split["folds"])):
        raise ValueError(f"fold must be in [0, {len(split['folds']) - 1}]")
    seed = config.training.seeds[0] if args.seed is None else args.seed
    if seed not in config.training.seeds:
        raise ValueError(f"seed must be one of {config.training.seeds}")
    all_jobs = [
        (fold, seed_value, config.training.epochs)
        for fold in range(len(split["folds"]))
        for seed_value in config.training.seeds
    ]
    if args.smoke:
        jobs = [(args.fold, seed, 1)]
    elif args.resume:
        jobs = [
            job for job in all_jobs
            if not (
                config.paths.output_dir
                / f"fold{job[0]}_seed{job[1]}"
                / "run_result.json"
            ).exists()
        ]
    else:
        jobs = all_jobs
    results = [
        train_one(dataset, split, config, fold, seed_value, device, epochs)
        for fold, seed_value, epochs in jobs
    ]
    if args.resume:
        results = [
            json.loads((
                config.paths.output_dir / f"fold{fold}_seed{seed_value}" / "run_result.json"
            ).read_text(encoding="utf-8"))
            for fold, seed_value, _ in all_jobs
        ]
    summary = {
        "schema": "AERIAL_LF_SUITE_V01",
        "status": "completed",
        "mode": "smoke" if args.smoke else ("resume" if args.resume else "full"),
        "runs": results,
    }
    if not args.smoke:
        summary["aggregate"] = aggregate_suite(config, dataset, split)
    (config.paths.output_dir / "run_index.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

