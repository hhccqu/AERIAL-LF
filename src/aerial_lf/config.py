"""Configuration loading for the public training pipeline."""
from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib


@dataclass(frozen=True)
class PathsConfig:
    features: Path
    split: Path
    semantic_registry: Path
    semantic_search_roots: tuple[Path, ...]
    baseline_checkpoint: str
    output_dir: Path

    def checkpoint(self, fold: int, seed: int) -> Path:
        return Path(self.baseline_checkpoint.format(fold=fold, seed=seed))


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (20260827, 20260828, 20260829)
    epochs: int = 50
    batch_size: int = 16
    fusion_lr: float = 1e-3
    head_lr: float = 2e-4
    weight_decay: float = 1e-4
    num_threads: int = 1
    device: str = "auto"


@dataclass(frozen=True)
class BaselineConfig:
    epochs: int = 50
    batch_size: int = 32
    learning_rate: float = 1e-3
    weight_decay: float = 1e-2


@dataclass(frozen=True)
class ExperimentConfig:
    source: Path
    paths: PathsConfig
    training: TrainingConfig
    baseline: BaselineConfig


def _expand(value: str, base: Path) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    return expanded if expanded.is_absolute() else (base / expanded).resolve()


def load_config(path: str | Path) -> ExperimentConfig:
    source = Path(path).resolve()
    with source.open("rb") as handle:
        raw = tomllib.load(handle)
    base = source.parent
    paths = raw["paths"]
    checkpoint = os.path.expandvars(os.path.expanduser(paths["baseline_checkpoint"]))
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint = str((base / checkpoint_path).resolve())
    path_config = PathsConfig(
        features=_expand(paths["features"], base),
        split=_expand(paths["split"], base),
        semantic_registry=_expand(paths["semantic_registry"], base),
        semantic_search_roots=tuple(
            _expand(item, base) for item in paths.get("semantic_search_roots", [])
        ),
        baseline_checkpoint=checkpoint,
        output_dir=_expand(paths["output_dir"], base),
    )
    training = TrainingConfig(**{
        **raw.get("training", {}),
        "seeds": tuple(raw.get("training", {}).get("seeds", TrainingConfig.seeds)),
    })
    baseline = BaselineConfig(**raw.get("baseline", {}))
    if not training.seeds:
        raise ValueError("training.seeds must not be empty")
    if training.epochs < 1 or training.batch_size < 1:
        raise ValueError("training epochs and batch_size must be positive")
    return ExperimentConfig(source, path_config, training, baseline)
