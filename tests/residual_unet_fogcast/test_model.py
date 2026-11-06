import pytest
import torch
from torch import nn
from fogcast import ResidualUNet
from fogcast.residual_unet_fogcast import ResidualBlock2d

def test_real_architecture_full_grid_forward_and_backward():
    previous = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        torch.manual_seed(42)
        model = ResidualUNet()
        assert sum((p.numel() for p in model.parameters())) == 2110538
        assert model.fuse.main[0].in_channels == 57
        assert [model.stem.main[0].out_channels, model.down1.main[0].out_channels, model.down2.main[0].out_channels, model.fuse.main[0].out_channels, model.down3.main[0].out_channels, model.bottleneck.main[0].out_channels] == [16, 32, 32, 64, 128, 256]
        assert all((m.num_groups == 8 and m.eps == 1e-05 for m in model.modules() if isinstance(m, nn.GroupNorm)))
        logits = model(torch.rand(1, 5, 600, 600), torch.rand(1, 25, 150, 150))
        assert logits.shape == (1, 10, 400, 400)
        assert torch.isfinite(logits).all()
        logits.square().mean().backward()
        assert all((p.grad is not None and torch.isfinite(p.grad).all() for p in model.parameters()))
    finally:
        torch.set_num_threads(previous)

def test_shape_and_nonfinite_inputs_rejected():
    model = ResidualUNet()
    with pytest.raises(ValueError, match='expected matched'):
        model(torch.zeros(1, 5, 40, 40), torch.zeros(1, 25, 10, 10))
    with pytest.raises(ValueError, match='expected matched'):
        model(torch.zeros(2, 5, 600, 600), torch.zeros(1, 25, 150, 150))
    with pytest.raises(ValueError, match='finite'):
        model(torch.full((1, 5, 600, 600), float('nan')), torch.zeros(1, 25, 150, 150))

def test_residual_shortcut_and_configuration_validation():
    assert isinstance(ResidualBlock2d(16, 16).shortcut, nn.Identity)
    projection = ResidualBlock2d(16, 32, 2)
    assert projection.main[0].stride == (2, 2)
    assert isinstance(projection.shortcut[0], nn.AvgPool2d)
    assert projection.shortcut[1].stride == (1, 1)
    assert projection(torch.rand(1, 16, 25, 25)).shape == (1, 32, 13, 13)
    with pytest.raises(ValueError):
        ResidualBlock2d(5, 7)
    with pytest.raises(ValueError):
        ResidualBlock2d(16, 32, stride=3)
