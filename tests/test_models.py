import pytest
import torch

from scramblemix import ScrambleMix, scramblemix_loss
from scramblemix.models import MODELS, ShakePyramidNet, WideResNet, build_model


@pytest.mark.parametrize("name", ["resnet18", "wrn40_2", "resnext29_16x4d"])
def test_forward_shapes(name):
    model = build_model(name, 100).eval()
    assert model(torch.rand(2, 3, 32, 32)).shape == (2, 100)


def test_registry_and_paper_model_sizes():
    assert set(MODELS) >= {"shakedrop", "wrn40_2", "wrn40_10", "resnet18", "resnext29_16x4d"}
    n = sum(p.numel() for p in build_model("shakedrop", 10).parameters())
    assert 28.4e6 < n < 28.6e6  # PyramidNet-110 (alpha 270)
    with pytest.raises(ValueError):
        build_model("vgg", 10)


def test_shakedrop_train_and_eval():
    torch.manual_seed(0)
    model = ShakePyramidNet(depth=20, alpha=48, num_classes=10)  # small PyramidNet for the test
    x = torch.rand(4, 3, 32, 32)
    out = model(x)
    out.sum().backward()
    assert out.shape == (4, 10)
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    model.eval()
    with torch.no_grad():
        assert torch.equal(model(x), model(x))  # ShakeDrop is deterministic at test time


def test_one_scramblemix_optimisation_step():
    """D = 2 views -> CE + self-teaching loss -> SGD step lowers the loss on the same batch."""
    torch.manual_seed(0)
    sm = ScrambleMix.random(num_pairs=2, key_seed=0, seed=0)
    images = torch.rand(8, 3, 32, 32)
    target = torch.randint(0, 10, (8,))
    views = sm.views(images)  # [B, D, 3, H, W]
    model = WideResNet(depth=10, num_classes=10, widen_factor=1)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)

    def loss_fn():
        return scramblemix_loss(torch.stack([model(views[:, d]) for d in range(2)]), target, lam=1.0)

    before = loss_fn()
    opt.zero_grad()
    before.backward()
    opt.step()
    after = loss_fn()
    assert torch.isfinite(before) and after.item() < before.item()
