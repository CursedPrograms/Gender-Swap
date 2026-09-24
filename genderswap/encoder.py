"""e4e (Encoder4Editing) image -> StyleGAN2 W+ encoder.

Adapted from https://github.com/omertov/encoder4editing (MIT).
"""
import math
from collections import namedtuple

from torch import nn
from torch.nn import functional as F

from .stylegan2 import EqualLinear


class Bottleneck(namedtuple('Block', ['in_channel', 'depth', 'stride'])):
    """A ResNet block description."""


def get_block(in_channel, depth, num_units, stride=2):
    return [Bottleneck(in_channel, depth, stride)] + [Bottleneck(depth, depth, 1) for _ in range(num_units - 1)]


def get_blocks_50():
    return [
        get_block(in_channel=64, depth=64, num_units=3),
        get_block(in_channel=64, depth=128, num_units=4),
        get_block(in_channel=128, depth=256, num_units=14),
        get_block(in_channel=256, depth=512, num_units=3),
    ]


class SEModule(nn.Module):
    def __init__(self, channels, reduction):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Conv2d(channels, channels // reduction, kernel_size=1, padding=0, bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(channels // reduction, channels, kernel_size=1, padding=0, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        w = self.sigmoid(self.fc2(self.relu(self.fc1(self.avg_pool(x)))))
        return x * w


class BottleneckIRSE(nn.Module):
    def __init__(self, in_channel, depth, stride):
        super().__init__()
        if in_channel == depth:
            self.shortcut_layer = nn.MaxPool2d(1, stride)
        else:
            self.shortcut_layer = nn.Sequential(
                nn.Conv2d(in_channel, depth, (1, 1), stride, bias=False),
                nn.BatchNorm2d(depth),
            )
        self.res_layer = nn.Sequential(
            nn.BatchNorm2d(in_channel),
            nn.Conv2d(in_channel, depth, (3, 3), (1, 1), 1, bias=False),
            nn.PReLU(depth),
            nn.Conv2d(depth, depth, (3, 3), stride, 1, bias=False),
            nn.BatchNorm2d(depth),
            SEModule(depth, 16),
        )

    def forward(self, x):
        return self.res_layer(x) + self.shortcut_layer(x)


def _upsample_add(x, y):
    _, _, h, w = y.size()
    return F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True) + y


class GradualStyleBlock(nn.Module):
    def __init__(self, in_c, out_c, spatial):
        super().__init__()
        self.out_c = out_c
        num_pools = int(math.log2(spatial))
        modules = [nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        for _ in range(num_pools - 1):
            modules += [nn.Conv2d(out_c, out_c, kernel_size=3, stride=2, padding=1), nn.LeakyReLU()]
        self.convs = nn.Sequential(*modules)
        self.linear = EqualLinear(out_c, out_c, lr_mul=1)

    def forward(self, x):
        x = self.convs(x)
        return self.linear(x.view(-1, self.out_c))


class Encoder4Editing(nn.Module):
    def __init__(self, stylegan_size=1024):
        super().__init__()
        self.input_layer = nn.Sequential(
            nn.Conv2d(3, 64, (3, 3), 1, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        modules = []
        for block in get_blocks_50():
            for b in block:
                modules.append(BottleneckIRSE(b.in_channel, b.depth, b.stride))
        self.body = nn.Sequential(*modules)

        self.style_count = 2 * int(math.log(stylegan_size, 2)) - 2
        self.coarse_ind = 3
        self.middle_ind = 7

        self.styles = nn.ModuleList()
        for i in range(self.style_count):
            if i < self.coarse_ind:
                spatial = 16
            elif i < self.middle_ind:
                spatial = 32
            else:
                spatial = 64
            self.styles.append(GradualStyleBlock(512, 512, spatial))

        self.latlayer1 = nn.Conv2d(256, 512, kernel_size=1, stride=1, padding=0)
        self.latlayer2 = nn.Conv2d(128, 512, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x = self.input_layer(x)
        for i, layer in enumerate(self.body):
            x = layer(x)
            if i == 6:
                c1 = x
            elif i == 20:
                c2 = x
            elif i == 23:
                c3 = x

        # Infer a base W, then per-layer deltas from progressively finer FPN features.
        w0 = self.styles[0](c3)
        w = w0.repeat(self.style_count, 1, 1).permute(1, 0, 2)
        features = c3
        for i in range(1, self.style_count):
            if i == self.coarse_ind:
                p2 = _upsample_add(c3, self.latlayer1(c2))
                features = p2
            elif i == self.middle_ind:
                p1 = _upsample_add(p2, self.latlayer2(c1))
                features = p1
            w[:, i] += self.styles[i](features)
        return w
