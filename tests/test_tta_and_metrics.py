import numpy as np
import torch
import torch.nn.functional as F

from scramblemix import ScrambleMix, average_predictions, lpips_distance, lpips_model, predict_tta


class Linear(torch.nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.fc = torch.nn.Linear(3 * 32 * 32, 10)

    def forward(self, x):
        return self.fc(x.flatten(1))


def test_average_predictions_logits_and_probs():
    model = Linear().eval()
    views = torch.rand(4, 3, 3, 32, 32)  # [B, T, 3, H, W]
    logits = torch.stack([model(views[:, t]) for t in range(3)])
    assert torch.allclose(average_predictions(model, views, "logits"), logits.mean(0), atol=1e-6)
    assert torch.allclose(average_predictions(model, views, "probs"), F.softmax(logits, -1).mean(0), atol=1e-6)
    assert torch.allclose(average_predictions(model, list(views.unbind(1))), logits.mean(0), atol=1e-6)


def test_predict_tta_uses_the_key_pairs():
    model = Linear().eval()
    sm = ScrambleMix.random(num_pairs=4, key_seed=0, mix_ratio=0.5)
    images = torch.rand(5, 3, 32, 32)
    out = predict_tta(model, images, sm, num_keys=2)
    assert out.shape == (5, 10)
    expected = (model(sm.mix(images, 0)) + model(sm.mix(images, 1))) / 2
    assert torch.allclose(out, expected, atol=1e-5)
    assert predict_tta(model, images[0], sm).shape == (1, 10)


def test_lpips_distance_offline_model():
    model = lpips_model("alex", "cpu", pretrained_backbone=False)  # random backbone: no download
    x = torch.rand(3, 3, 32, 32)
    d = lpips_distance(x, x, model=model)
    assert d.shape == (3,) and torch.allclose(d, torch.zeros(3), atol=1e-6)
    y = ScrambleMix.random(num_pairs=1, key_seed=0)(x)
    assert (lpips_distance(x, y, model=model) > 0).all()
    u8 = (x * 255).round().to(torch.uint8)
    assert lpips_distance(u8[0], np.asarray(u8[0].permute(1, 2, 0)), model=model).shape == (1,)
