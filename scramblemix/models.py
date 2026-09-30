"""CIFAR-size classifiers used with ScrambleMix.

The networks are device-agnostic rewrites of the ones in the original code; they return logits
only (the original ones returned ``(logits, features)``). Parameter names are unchanged, and the
final fixed-size average pooling is written as global average pooling (identical for 32x32 inputs).

* ``shakedrop`` -- PyramidNet-110 (alpha = 270) with ShakeDrop, the "Shakedrop" rows of slides
  39-40. From ``archive/models/no_adaptation_network.py`` + ``archive/models/shakedrop.py``,
  based on https://github.com/owruby/shake-drop_pytorch (ShakeDrop: Yamada et al., 2019).
* ``wrn40_2`` -- Wide ResNet built exactly as the original code builds its ``wideresnet`` model
  (``WideResNet(num_classes=...)`` with the defaults ``depth=40, widen_factor=2``) from
  ``archive/scripts/scramblemix/wideresnet.py`` (https://github.com/xternalz/WideResNet-pytorch,
  as bundled with AugMix). The slides label this network "WideResNet40x10"; ``wrn40_10`` builds
  the widen-factor-10 variant.
* ``resnext29_16x4d`` -- ResNeXt-29 (16x4d), the ``senet2`` model of ``main_paper.sh``
  (``archive/scripts/scramblemix/resnext2.py``, from https://github.com/kuangliu/pytorch-cifar).
* ``resnet18`` -- CIFAR ResNet-18 (https://github.com/kuangliu/pytorch-cifar), a light model for
  quick experiments. Note: in the original ``main.py`` the name ``resnet18`` fell through to the
  ShakeDrop PyramidNet.
"""
from __future__ import annotations

import math
from typing import Callable, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


# ----------------------------------------------------------------------------- ShakeDrop PyramidNet
class ShakeDropFunction(torch.autograd.Function):
    """ShakeDrop (Yamada et al., 2019) as implemented by owruby/shake-drop_pytorch: one Bernoulli gate
    per mini-batch; when the gate is off the branch is scaled by alpha ~ U(-1, 1) per sample in the
    forward pass and the gradient by beta ~ U(0, 1) per sample in the backward pass."""

    @staticmethod
    def forward(ctx, x, training=True, p_drop=0.5, alpha_range=(-1.0, 1.0)):
        if not training:
            ctx.gate, ctx.scale = None, 1.0 - p_drop
            return (1.0 - p_drop) * x
        ctx.gate = bool(torch.rand(()) < 1.0 - p_drop)
        if ctx.gate:
            return x.clone()
        alpha = torch.empty(x.size(0), 1, 1, 1, device=x.device, dtype=x.dtype).uniform_(*alpha_range)
        return alpha * x

    @staticmethod
    def backward(ctx, grad_output):
        if ctx.gate is None:
            return grad_output * ctx.scale, None, None, None
        if ctx.gate:
            return grad_output, None, None, None
        beta = torch.empty(grad_output.size(0), 1, 1, 1, device=grad_output.device,
                           dtype=grad_output.dtype).uniform_(0.0, 1.0)
        return beta * grad_output, None, None, None


class ShakeDrop(nn.Module):
    def __init__(self, p_drop: float = 0.5, alpha_range=(-1.0, 1.0)) -> None:
        super().__init__()
        self.p_drop = p_drop
        self.alpha_range = tuple(alpha_range)

    def forward(self, x):
        return ShakeDropFunction.apply(x, self.training, self.p_drop, self.alpha_range)


class ShakeBasicBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int = 1, p_shakedrop: float = 1.0) -> None:
        super().__init__()
        self.downsampled = stride == 2
        self.branch = nn.Sequential(
            nn.BatchNorm2d(in_ch),
            nn.Conv2d(in_ch, out_ch, 3, padding=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, stride=1, bias=False),
            nn.BatchNorm2d(out_ch))
        self.shortcut = nn.AvgPool2d(2)  # only applied when downsampling
        self.shake_drop = ShakeDrop(p_shakedrop)

    def forward(self, x):
        h = self.shake_drop(self.branch(x))
        h0 = self.shortcut(x) if self.downsampled else x
        pad = h0.new_zeros(h0.size(0), h.size(1) - h0.size(1), h0.size(2), h0.size(3))
        return h + torch.cat([h0, pad], dim=1)


class ShakePyramidNet(nn.Module):
    """PyramidNet with ShakeDrop; ``depth=110, alpha=270`` in the paper's experiments."""

    def __init__(self, depth: int = 110, alpha: int = 270, num_classes: int = 10) -> None:
        super().__init__()
        if (depth - 2) % 6:
            raise ValueError("depth must be 6n + 2")
        in_ch = 16
        n_units = (depth - 2) // 6
        in_chs = [in_ch] + [in_ch + math.ceil((alpha / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]
        self.in_chs, self.u_idx = in_chs, 0
        # drop probability grows linearly from 0.5 / (3n) to 0.5 with depth
        self.ps_shakedrop = [1 - (1.0 - (0.5 / (3 * n_units)) * (i + 1)) for i in range(3 * n_units)]

        self.c_in = nn.Conv2d(3, in_chs[0], 3, padding=1)
        self.bn_in = nn.BatchNorm2d(in_chs[0])
        self.layer1 = self._make_layer(n_units, 1)
        self.layer2 = self._make_layer(n_units, 2)
        self.layer3 = self._make_layer(n_units, 2)
        self.bn_out = nn.BatchNorm2d(in_chs[-1])
        self.fc_out = nn.Linear(in_chs[-1], num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def _make_layer(self, n_units: int, stride: int) -> nn.Sequential:
        layers = []
        for _ in range(n_units):
            layers.append(ShakeBasicBlock(self.in_chs[self.u_idx], self.in_chs[self.u_idx + 1],
                                          stride, self.ps_shakedrop[self.u_idx]))
            self.u_idx, stride = self.u_idx + 1, 1
        return nn.Sequential(*layers)

    def forward(self, x):
        h = self.bn_in(self.c_in(x))
        h = self.layer3(self.layer2(self.layer1(h)))
        h = F.relu(self.bn_out(h))
        h = F.adaptive_avg_pool2d(h, 1).flatten(1)
        return self.fc_out(h)


# ----------------------------------------------------------------------------- Wide ResNet
class WideBasicBlock(nn.Module):
    def __init__(self, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_planes, out_planes, 3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_planes, out_planes, 3, stride=1, padding=1, bias=False)
        self.droprate = drop_rate
        self.equalInOut = in_planes == out_planes
        self.convShortcut = None if self.equalInOut else nn.Conv2d(in_planes, out_planes, 1, stride=stride,
                                                                   padding=0, bias=False)

    def forward(self, x):
        o = self.relu1(self.bn1(x))
        if not self.equalInOut:
            x = o  # the shortcut uses the pre-activated input when the width changes
        out = self.relu2(self.bn2(self.conv1(o)))
        if self.droprate > 0:
            out = F.dropout(out, p=self.droprate, training=self.training)
        out = self.conv2(out)
        return (x if self.equalInOut else self.convShortcut(x)) + out


class NetworkBlock(nn.Module):
    def __init__(self, nb_layers: int, in_planes: int, out_planes: int, stride: int, drop_rate: float = 0.0):
        super().__init__()
        self.layer = nn.Sequential(*[
            WideBasicBlock(in_planes if i == 0 else out_planes, out_planes, stride if i == 0 else 1, drop_rate)
            for i in range(nb_layers)])

    def forward(self, x):
        return self.layer(x)


class WideResNet(nn.Module):
    def __init__(self, depth: int = 40, num_classes: int = 10, widen_factor: int = 2, drop_rate: float = 0.0):
        super().__init__()
        if (depth - 4) % 6:
            raise ValueError("depth must be 6n + 4")
        ch = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        n = (depth - 4) // 6
        self.conv1 = nn.Conv2d(3, ch[0], 3, stride=1, padding=1, bias=False)
        self.block1 = NetworkBlock(n, ch[0], ch[1], 1, drop_rate)
        self.block2 = NetworkBlock(n, ch[1], ch[2], 2, drop_rate)
        self.block3 = NetworkBlock(n, ch[2], ch[3], 2, drop_rate)
        self.bn1 = nn.BatchNorm2d(ch[3])
        self.relu = nn.ReLU(inplace=True)
        self.fc = nn.Linear(ch[3], num_classes)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        out = self.block3(self.block2(self.block1(self.conv1(x))))
        out = self.relu(self.bn1(out))
        return self.fc(F.adaptive_avg_pool2d(out, 1).flatten(1))


# ----------------------------------------------------------------------------- ResNet-18 / ResNeXt-29
class BasicBlock(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))


class ResNet(nn.Module):
    def __init__(self, num_blocks=(2, 2, 2, 2), num_classes: int = 10) -> None:
        super().__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(64, num_blocks[0], 1)
        self.layer2 = self._make_layer(128, num_blocks[1], 2)
        self.layer3 = self._make_layer(256, num_blocks[2], 2)
        self.layer4 = self._make_layer(512, num_blocks[3], 2)
        self.linear = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(BasicBlock(self.in_planes, planes, s))
            self.in_planes = planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        return self.linear(F.adaptive_avg_pool2d(out, 1).flatten(1))


class ResNeXtBlock(nn.Module):
    expansion = 2

    def __init__(self, in_planes: int, cardinality: int = 32, bottleneck_width: int = 4, stride: int = 1):
        super().__init__()
        gw = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, gw, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(gw)
        self.conv2 = nn.Conv2d(gw, gw, 3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(gw)
        self.conv3 = nn.Conv2d(gw, self.expansion * gw, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * gw)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * gw:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, self.expansion * gw, 1, stride=stride, bias=False),
                                          nn.BatchNorm2d(self.expansion * gw))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        return F.relu(out + self.shortcut(x))


class ResNeXt(nn.Module):
    def __init__(self, num_blocks=(3, 3, 3), cardinality: int = 16, bottleneck_width: int = 4,
                 num_classes: int = 10) -> None:
        super().__init__()
        self.cardinality, self.bottleneck_width, self.in_planes = cardinality, bottleneck_width, 64
        self.conv1 = nn.Conv2d(3, 64, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(ResNeXtBlock(self.in_planes, self.cardinality, self.bottleneck_width, s))
            self.in_planes = ResNeXtBlock.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer3(self.layer2(self.layer1(out)))
        return self.linear(F.adaptive_avg_pool2d(out, 1).flatten(1))


# ----------------------------------------------------------------------------- registry
def shake_pyramidnet(num_classes: int = 10, depth: int = 110, alpha: int = 270) -> ShakePyramidNet:
    return ShakePyramidNet(depth=depth, alpha=alpha, num_classes=num_classes)


def wrn40_2(num_classes: int = 10) -> WideResNet:
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=2)


def wrn40_10(num_classes: int = 10) -> WideResNet:
    return WideResNet(depth=40, num_classes=num_classes, widen_factor=10)


def resnet18(num_classes: int = 10) -> ResNet:
    return ResNet((2, 2, 2, 2), num_classes)


def resnext29_16x4d(num_classes: int = 10) -> ResNeXt:
    return ResNeXt((3, 3, 3), cardinality=16, bottleneck_width=4, num_classes=num_classes)


MODELS: Dict[str, Callable[[int], nn.Module]] = {
    "shakedrop": shake_pyramidnet,
    "wrn40_2": wrn40_2,
    "wrn40_10": wrn40_10,
    "resnext29_16x4d": resnext29_16x4d,
    "resnet18": resnet18,
}


def build_model(name: str, num_classes: int) -> nn.Module:
    """Instantiate one of :data:`MODELS` by name."""
    if name not in MODELS:
        raise ValueError(f"unknown model {name!r}; choose from {sorted(MODELS)}")
    return MODELS[name](num_classes)
