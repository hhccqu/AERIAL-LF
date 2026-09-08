import numpy as np

from aerial_lf.dino_lift import position_encoding
from aerial_lf.geometry import pnp_quality_gate


def test_archived_pnp_gate():
    values = pnp_quality_gate(np.asarray([0, 5, 7.5, 10, 12], np.float32))
    np.testing.assert_array_equal(values, np.asarray([1, 1, 0.5, 0, 0], np.float32))


def test_position_encoding_contract():
    encoded = position_encoding(np.zeros((36, 2), np.float32))
    assert encoded.shape == (36, 768)
    assert encoded.dtype == np.float32
    assert np.isfinite(encoded).all()

