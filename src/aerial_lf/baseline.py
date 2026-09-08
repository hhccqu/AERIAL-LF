"""Vehicle-only baseline training used to initialize AERIAL-LF."""
from __future__ import annotations

import argparse
import copy
import json
import os
import random

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch
from torch import nn

from .config import load_config
from .data import FeatureDataset, load_feature_archive, load_split, selected_branch_loss
from .evaluate import waypoint_metrics
from .model import Head
from .provenance import sha256
from .train import select_device


class VehicleOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.head = Head()

    def forward(self, scene_tokens):
        return self.head(scene_tokens)


def seed_all(value: int) -> None:
    random.seed(value)
    np.random.seed(value)
    torch.manual_seed(value)
    torch.use_deterministic_algorithms(True, warn_only=True)


def _batch(dataset: FeatureDataset, indices: np.ndarray, device: torch.device):
    rows = dataset.scene_row[indices]
    return (
        torch.as_tensor(dataset.arrays["scene_query"][rows], device=device),
        torch.as_tensor(dataset.arrays["route_target_6x2"][indices], device=device),
        torch.as_tensor(dataset.arrays["route_branch"][indices], device=device),
    )


def train_one(dataset, split, config, fold, seed, device):
    params = config.baseline
    seed_all(seed)
    fold_spec = split["folds"][fold]
    train_indices = dataset.route_indices(fold_spec["train_scenes"])
    val_indices = dataset.route_indices(fold_spec["val_scenes"])
    test_indices = dataset.route_indices(fold_spec["test_scenes"])
    model = VehicleOnlyModel().to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=params.learning_rate, weight_decay=params.weight_decay
    )
    best_loss = float("inf")
    best_state = None
    best_epoch = None
    history = []
    for epoch in range(1, params.epochs + 1):
        model.train()
        order = np.random.default_rng(seed + epoch).permutation(train_indices)
        losses = []
        for start in range(0, len(order), params.batch_size):
            scene, target, branch = _batch(dataset, order[start:start + params.batch_size], device)
            current = selected_branch_loss(model(scene), target, branch)
            optimizer.zero_grad(set_to_none=True)
            current.backward()
            optimizer.step()
            losses.append(float(current.detach()))
        model.eval()
        with torch.no_grad():
            scene, target, branch = _batch(dataset, val_indices, device)
            val_loss = float(selected_branch_loss(model(scene), target, branch))
        history.append({
            "epoch": epoch,
            "train_l1_m": float(np.mean(losses)),
            "val_l1_m": val_loss,
        })
        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
    if best_state is None:
        raise RuntimeError("baseline training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        scene, target, branch = _batch(dataset, test_indices, device)
        prediction = model(scene).cpu().numpy()
    target_np = dataset.arrays["route_target_6x2"][test_indices]
    branch_np = dataset.arrays["route_branch"][test_indices]
    checkpoint = config.paths.checkpoint(fold, seed)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), checkpoint)
    result = {
        "schema": "AERIAL_LF_VEHICLE_BASELINE_RUN_V01",
        "status": "completed",
        "fold": fold,
        "seed": seed,
        "best_epoch": best_epoch,
        "best_val_l1_m": best_loss,
        "metrics": waypoint_metrics(prediction, target_np, branch_np),
        "training": {
            "epochs": params.epochs,
            "batch_size": params.batch_size,
            "learning_rate": params.learning_rate,
            "weight_decay": params.weight_decay,
        },
        "inputs": {
            "feature_sha256": sha256(config.paths.features),
            "split_sha256": sha256(config.paths.split),
            "config_sha256": sha256(config.source),
        },
    }
    checkpoint.with_suffix(".json").write_text(
        json.dumps({**result, "history": history}, indent=2) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--fold", type=int)
    parser.add_argument("--seed", type=int)
    args = parser.parse_args()
    config = load_config(args.config)
    torch.set_num_threads(config.training.num_threads)
    device = select_device(config.training.device)
    arrays, scene_row = load_feature_archive(config.paths.features)
    dataset = FeatureDataset(arrays, {}, scene_row)
    split = load_split(config.paths.split, dataset)
    folds = range(len(split["folds"])) if args.fold is None else [args.fold]
    seeds = config.training.seeds if args.seed is None else [args.seed]
    results = [
        train_one(dataset, split, config, fold, seed, device)
        for fold in folds for seed in seeds
    ]
    print(json.dumps({"status": "completed", "runs": results}, indent=2))


if __name__ == "__main__":
    main()

