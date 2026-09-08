import importlib.util
import os
from pathlib import Path

import pytest
import torch

from aerial_lf.model import F15JointHeadModel


def test_archived_f15_equivalence():
    source = os.environ.get("AERIAL_LF_ARCHIVED_MODEL")
    if not source:
        pytest.skip("set AERIAL_LF_ARCHIVED_MODEL to enable archive comparison")
    path = Path(source)
    specification = importlib.util.spec_from_file_location("archived_f15", path)
    module = importlib.util.module_from_spec(specification)
    assert specification.loader is not None
    specification.loader.exec_module(module)
    torch.manual_seed(123)
    archived = module.F15JointHeadModel().eval()
    torch.manual_seed(123)
    released = F15JointHeadModel().eval()
    assert list(archived.state_dict()) == list(released.state_dict())
    for key in archived.state_dict():
        torch.testing.assert_close(
            archived.state_dict()[key], released.state_dict()[key], rtol=0, atol=0
        )
    values = (
        torch.randn(1, 16, 256),
        torch.randn(1, 36, 768),
        torch.ones(1, 36, dtype=torch.bool),
        torch.randn(1, 36, 2),
        torch.randn(1, 2, 32, 64),
        torch.randn(1, 32, 64, 2),
        torch.zeros(1, 32, 64, dtype=torch.bool),
        torch.zeros(1, 32, 64, dtype=torch.bool),
        torch.tensor([0.75]),
    )
    with torch.no_grad():
        old_prediction, old_diagnostics = archived(*values)
        new_prediction, new_diagnostics = released(*values)
    torch.testing.assert_close(old_prediction, new_prediction, rtol=0, atol=0)
    torch.testing.assert_close(
        old_diagnostics["semantic_token_mask"],
        new_diagnostics["semantic_token_mask"],
        rtol=0,
        atol=0,
    )

