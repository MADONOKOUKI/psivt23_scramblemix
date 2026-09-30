import pytest
import torch
import torch.nn.functional as F

from scramblemix import scramblemix_loss, self_teaching_loss


def original_objective(outputs_list, targets, num_views, js=True):
    """The loss of the original code, transcribed from archive/scripts/scramblemix/trainer.py (train())."""
    p_mixture = 0
    for i in range(num_views):
        p_mixture = p_mixture + F.softmax(outputs_list[i], dim=1)
    p_mixture = (p_mixture / num_views).log()
    loss_js = 0
    for i in range(num_views):
        loss_js = loss_js + F.kl_div(p_mixture, F.softmax(outputs_list[i], dim=1), reduction="batchmean")
    loss_js = loss_js / num_views
    loss = 0
    for i in range(num_views):
        loss = loss + torch.nn.CrossEntropyLoss()(outputs_list[i], targets)
    loss = loss / num_views
    return loss + loss_js if js else loss


@pytest.mark.parametrize("d", [1, 2, 4])
def test_matches_original_objective(d):
    torch.manual_seed(d)
    logits = torch.randn(d, 16, 10) * 3
    target = torch.randint(0, 10, (16,))
    ours = scramblemix_loss(logits, target, lam=1.0)
    ref = original_objective(list(logits), target, d)
    assert torch.allclose(ours, ref, atol=1e-5)
    assert torch.allclose(scramblemix_loss(logits, target, lam=0.0), original_objective(list(logits), target, d, js=False))


def test_stop_gradient_does_not_change_the_gradient():
    """Slide 35 detaches the average posterior; the original code does not. Same gradients either way."""
    torch.manual_seed(0)
    base = torch.randn(4, 8, 10) * 2
    target = torch.randint(0, 10, (8,))
    grads = []
    for fn in (lambda z: scramblemix_loss(z, target, stop_grad=True),
               lambda z: scramblemix_loss(z, target, stop_grad=False),
               lambda z: original_objective(list(z), target, 4)):
        z = base.clone().requires_grad_(True)
        fn(z).backward()
        grads.append(z.grad)
    assert torch.allclose(grads[0], grads[1], atol=1e-6)
    assert torch.allclose(grads[0], grads[2], atol=1e-6)


def test_self_teaching_loss_properties():
    z = torch.randn(1, 5, 7)
    assert self_teaching_loss(z).item() == pytest.approx(0.0, abs=1e-7)  # D = 1
    same = torch.randn(1, 5, 7).expand(3, 5, 7)
    assert self_teaching_loss(same).item() == pytest.approx(0.0, abs=1e-6)  # identical views
    assert self_teaching_loss(torch.randn(3, 5, 7)).item() > 0
    # list input == stacked input; extreme logits stay finite (computed in log space)
    views = [torch.randn(5, 7) for _ in range(3)]
    assert torch.allclose(self_teaching_loss(views), self_teaching_loss(torch.stack(views)))
    extreme = torch.tensor([[[500.0, 0.0]], [[0.0, 500.0]]])
    assert torch.isfinite(self_teaching_loss(extreme))


def test_rejects_bad_shapes():
    with pytest.raises(ValueError):
        self_teaching_loss(torch.randn(5, 7))
