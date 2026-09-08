# Pipeline and Ownership Boundaries

## 1. Physical Capture

Record synchronized ground-camera observations, one UAV image, vehicle pose,
UAV calibration, navigation metadata, and immutable scene identifiers.

## 2. Geometry

Estimate the scene homography and body pose from AprilTags. Store reprojection
RMS and the fixed body grid. Geometry associates aerial evidence with the
vehicle planning region; it is not optimized by waypoint loss.

## 3. Frozen Features

- SSR owner: export one `scene_query` per scene with the documented navigation
  metadata and checkpoint hash.
- DINOv2 owner: export patch features and the exact resize/padding manifest.
- SAM3 owner: export drivable/obstacle candidates, then complete the declared
  review process and retain invalid/unknown regions.

The public fusion trainer does not download or modify these backbones.

## 4. Independent Route References

Create six body-frame waypoint anchors per command-conditioned route. Route
references must not be generated from the semantic maps used as model input.
Record the annotation protocol and reviewer identity in the private dataset
documentation.

## 5. Assembly and Split

Join only by exact scene ID, validate every shape, hash source artifacts, and
assign all routes from a scene to one partition. Related scene configurations
that must not cross partitions share one `route_group_id`.

## 6. Training

Train fold/seed-specific vehicle-only heads, initialize AERIAL-LF from those
heads, and select checkpoints using validation loss only. Test predictions are
written once per run and aggregated by immutable `route_index`.

## 7. Reporting

Keep single-seed metrics distinct from the prediction ensemble. Report overall
and maneuver-level behavior, route counts, data version hashes, and any
controlled blocked/unblocked analysis as separate evidence.

