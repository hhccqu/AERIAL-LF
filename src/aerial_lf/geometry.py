"""Geometry-derived reliability functions used by the archived F15 model."""
from __future__ import annotations

import numpy as np


def pnp_quality_gate(pnp_rms_px: np.ndarray) -> np.ndarray:
    """Return the exact fixed F15 gate: clip((10 - RMS) / 5, 0, 1)."""
    values = np.asarray(pnp_rms_px, dtype=np.float32)
    return np.clip((10.0 - values) / 5.0, 0.0, 1.0).astype(np.float32)

