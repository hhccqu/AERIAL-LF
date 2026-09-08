# Data Contracts

All joins use exact `scene_id` strings. Directory order is never a data key.

## Feature NPZ

| Key | Shape | Meaning |
| --- | --- | --- |
| `scene_id` | `[S]` | Unique scene identifiers |
| `scene_query` | `[S,16,256]` | Frozen SSR planning tokens |
| `uav_tokens` | `[S,36,768]` | Frozen, lifted DINOv2 tokens |
| `uav_token_valid_mask` | `[S,36]` | Valid appearance-token support |
| `uav_token_anchor_ego_m` | `[S,36,2]` | Mean body-frame token anchors |
| `route_scene_id` | `[R]` | Scene identifier for each route |
| `route_command` | `[R]` | Human-readable command |
| `route_branch` | `[R]` | `0=right`, `1=left`, `2=straight` |
| `route_target_6x2` | `[R,6,2]` | Body-frame waypoint references in meters |
| `route_pnp_rms_px` | `[R]` | Scene geometry reprojection RMS |
| `route_group_id` | `[R]` | Leakage-control group identifier |

Additional scene-level arrays, such as `latent_pos`, are preserved by the
assembly utility but are not consumed by the F15 forward pass.

## Semantic NPZ

Each scene record referenced by the semantic registry contains:

| Key | Shape |
| --- | --- |
| `reviewed_candidate_drivable` | `[H,W]` |
| `reviewed_candidate_obstacle` | `[H,W]` |
| `invalid` | `[H,W]` |
| `unknown_reviewed` | `[H,W]` |
| `grid_body_xy_m` | `[H,W,2]` |

The archived release uses `H=180`, `W=360`. The semantic CNN accepts compatible
sizes, but all scenes in one minibatch must share a shape. A pooled semantic
token participates in attention when at least half of its support is neither
invalid nor unknown.

## Split JSON

The top-level `folds` list contains `train_scenes`, `val_scenes`, and
`test_scenes` for every fold. The loader verifies:

- scene partitions are disjoint and cover the complete scene set;
- route groups do not cross partitions;
- every scene appears in the held-out test partition exactly once.

## DINO Lift Registry

Each record names `scene_id`, `patch_npz`, `patch_manifest`, `geometry`, and
`grid_npz`. Paths may be absolute or relative to the registry. The patch NPZ
must contain `patch_features [1,37,37,768]`; the manifest stores the exact
resize/pad transform; geometry stores the world-to-image homography and vehicle
body pose.

