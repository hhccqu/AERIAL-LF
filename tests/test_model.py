import torch

from aerial_lf.data import selected_branch_loss
from aerial_lf.model import F15JointHeadModel, trainable_parameter_count


def inputs(batch=2):
    return (
        torch.randn(batch, 16, 256),
        torch.randn(batch, 36, 768),
        torch.ones(batch, 36, dtype=torch.bool),
        torch.randn(batch, 36, 2),
        torch.randn(batch, 2, 32, 64),
        torch.randn(batch, 32, 64, 2),
        torch.zeros(batch, 32, 64, dtype=torch.bool),
        torch.zeros(batch, 32, 64, dtype=torch.bool),
        torch.ones(batch),
    )


def test_model_contract_and_parameter_count():
    torch.manual_seed(7)
    model = F15JointHeadModel().eval()
    with torch.no_grad():
        prediction, diagnostics = model(*inputs())
    assert prediction.shape == (2, 3, 6, 2)
    assert diagnostics["semantic_token_mask"].shape == (2, 32)
    assert torch.all(torch.diff(prediction[..., 0], dim=-1) >= 0)
    assert trainable_parameter_count(model) == 523300


def test_zero_gate_preserves_vehicle_latent_for_decoding():
    torch.manual_seed(9)
    model = F15JointHeadModel().eval()
    values = list(inputs(batch=1))
    values[-1] = torch.zeros(1)
    with torch.no_grad():
        prediction, _ = model(*values)
        expected = model.head(values[0])
    torch.testing.assert_close(prediction, expected, rtol=0, atol=0)


def test_loss_supervises_only_selected_branch():
    prediction = torch.zeros(2, 3, 6, 2, requires_grad=True)
    target = torch.ones(2, 6, 2)
    branch = torch.tensor([0, 2])
    selected_branch_loss(prediction, target, branch).backward()
    assert prediction.grad[0, 0].abs().sum() > 0
    assert prediction.grad[1, 2].abs().sum() > 0
    assert prediction.grad[0, 1:].abs().sum() == 0
    assert prediction.grad[1, :2].abs().sum() == 0

