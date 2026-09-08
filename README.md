# AERIAL-LF

This directory is the behavior-preserving research release of the F15
AERIAL-LF pipeline used for the ICRA manuscript. It separates public code from
private sensor data, cached features, semantic annotations, and checkpoints.

## Model

```text
frozen SSR scene tokens [B,16,256]
  -> cross-attend to 36 frozen DINOv2 tokens with body-frame anchors
  -> bounded residual update
  -> cross-attend to dense SAM3-derived semantic tokens with body-frame positions
  -> bounded residual update
  -> mean pooling and a three-branch, six-waypoint head
```

The appearance branch is applied before the semantic branch. Only the route's
selected command branch receives L1 waypoint supervision. SSR, DINOv2, and
SAM3 outputs are cached and fixed during fusion training. The PnP term is the
archived fixed reliability scale `clip((10 - RMS_px) / 5, 0, 1)`; it is not a
learned module.

## Install

```bash
python -m venv .venv
.venv/Scripts/activate
python -m pip install -e ".[dev]"
```

On Linux or macOS, activate the environment with `source .venv/bin/activate`.

## Private Inputs

Copy `configs/f15_archive.example.toml` to a local config and point it at:

- one feature NPZ following `docs/DATA_CONTRACTS.md`;
- one scene/group-disjoint split JSON;
- one dense semantic registry;
- fold- and seed-specific vehicle-only baseline checkpoints.

Keep these files under `data/private/` or outside the repository. The release
does not contain raw images, human annotations, pretrained backbone weights,
private network addresses, or experiment checkpoints.

## Run

```bash
aerial-lf-audit --config configs/local.toml --require-checkpoints
aerial-lf-baseline --config configs/local.toml
aerial-lf-train --config configs/local.toml --smoke
aerial-lf-train --config configs/local.toml
```

Use `--resume` to train only missing fold/seed runs and then aggregate the full
OOF suite. Prediction ensembling averages the three predictions for each route
before ADE/FDE are computed, matching the archived F15 protocol.

## Expanding the Dataset

1. Export one frozen SSR `scene_query [16,256]` per new physical scene.
2. Export frozen DINOv2 patch features, then run `aerial-lf-dino-lift` with the
   accepted body-grid geometry.
3. Produce reviewed drivable/obstacle maps with invalid and unknown masks.
4. Add expert route references and command branches independently of SAM3.
5. Build a feature NPZ and semantic registry with exact scene-ID joins.
6. Create a new scene/group-disjoint split. Never split routes from one scene.
7. Train new vehicle-only baselines, audit inputs, and run AERIAL-LF.

`aerial-lf-assemble merge` can concatenate compatible old/new feature archives.
It rejects duplicate scene IDs and shape mismatches.

## Verification

The test suite checks output shapes, monotone forward coordinates, zero-gate
identity behavior, selected-branch supervision, split leakage, aggregation,
and optional equivalence with the archived F15 source.

```bash
pytest
```

Set `AERIAL_LF_ARCHIVED_MODEL` to the archived `f15_joint_head_model.py` path to
enable the cross-source equivalence test. See `docs/ARCHIVED_F15.md` for the
recorded result and provenance boundary.

Before public release, replace the anonymous author entry in `CITATION.cff` and
add the final paper DOI or URL.

