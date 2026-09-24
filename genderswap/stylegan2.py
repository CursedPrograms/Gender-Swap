"""StyleGAN2 generator (rosinality layout, as used by e4e), with pure-PyTorch ops.

The original code JIT-compiles CUDA kernels for upfirdn2d and fused_leaky_relu, which
is painful on Windows. These native versions are numerically equivalent and load the
same checkpoints.

Adapted from https://github.com/rosinality/stylegan2-pytorch (MIT) via
https://github.com/omertov/encoder4editing (MIT).
"""
import math

import torch
from torch import nn
from torch.nn import functional as F


def fused_leaky_relu(input, bias, negative_slope=0.2, scale=2 ** 0.5):
    shape = [1, -1] + [1] * (input.ndim - 2)
    return F.leaky_relu(input + bias.view(shape), negative_slope) * scale


class FusedLeakyReLU(nn.Module):
    def __init__(self, channel, negative_slope=0.2, scale=2 ** 0.5):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(channel))
        self.negative_slope = negative_slope
        self.scale = scale

    def forward(self, input):
        return fused_leaky_relu(input, self.bias, self.negative_slope, self.scale)


def upfirdn2d(input, kernel, up=1, down=1, pad=(0, 0)):
    """Upsample (zero-insert), FIR filter, downsample. NCHW, same pad on both axes."""
    pad0, pad1 = pad
    batch, channel, in_h, in_w = input.shape
    kernel_h, kernel_w = kernel.shape

    out = input
    if up > 1:
        out = out.reshape(batch, channel, in_h, 1, in_w, 1)
        out = F.pad(out, [0, up - 1, 0, 0, 0, up - 1])
        out = out.reshape(batch, channel, in_h * up, in_w * up)

    out = F.pad(out, [max(pad0, 0), max(pad1, 0), max(pad0, 0), max(pad1, 0)])
    out = out[
        :,
        :,
        max(-pad0, 0): out.shape[2] - max(-pad1, 0),
        max(-pad0, 0): out.shape[3] - max(-pad1, 0),
    ]

    weight = torch.flip(kernel, [0, 1]).view(1, 1, kernel_h, kernel_w)
    weight = weight.repeat(channel, 1, 1, 1).to(out.dtype)
    out = F.conv2d(out, weight, groups=channel)

    return out[:, :, ::down, ::down]


class PixelNorm(nn.Module):
    def forward(self, input):
        return input * torch.rsqrt(torch.mean(input ** 2, dim=1, keepdim=True) + 1e-8)


def make_kernel(k):
    k = torch.tensor(k, dtype=torch.float32)
    if k.ndim == 1:
        k = k[None, :] * k[:, None]
    k /= k.sum()
    return k


class Upsample(nn.Module):
    def __init__(self, kernel, factor=2):
        super().__init__()
        self.factor = factor
        kernel = make_kernel(kernel) * (factor ** 2)
        self.register_buffer('kernel', kernel)
        p = kernel.shape[0] - factor
        self.pad = ((p + 1) // 2 + factor - 1, p // 2)

    def forward(self, input):
        return upfirdn2d(input, self.kernel, up=self.factor, down=1, pad=self.pad)


class Blur(nn.Module):
    def __init__(self, kernel, pad, upsample_factor=1):
        super().__init__()
        kernel = make_kernel(kernel)
        if upsample_factor > 1:
            kernel = kernel * (upsample_factor ** 2)
        self.register_buffer('kernel', kernel)
        self.pad = pad

    def forward(self, input):
        return upfirdn2d(input, self.kernel, pad=self.pad)


class EqualLinear(nn.Module):
    def __init__(self, in_dim, out_dim, bias=True, bias_init=0, lr_mul=1, activation=None):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(out_dim, in_dim).div_(lr_mul))
        self.bias = nn.Parameter(torch.zeros(out_dim).fill_(bias_init)) if bias else None
        self.activation = activation
        self.scale = (1 / math.sqrt(in_dim)) * lr_mul
        self.lr_mul = lr_mul

    def forward(self, input):
        if self.activation:
            out = F.linear(input, self.weight * self.scale)
            return fused_leaky_relu(out, self.bias * self.lr_mul)
        return F.linear(input, self.weight * self.scale, bias=self.bias * self.lr_mul)


class ModulatedConv2d(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, demodulate=True,
                 upsample=False, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        self.kernel_size = kernel_size
        self.in_channel = in_channel
        self.out_channel = out_channel
        self.upsample = upsample

        if upsample:
            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1
            self.blur = Blur(blur_kernel, pad=(pad0, pad1), upsample_factor=factor)

        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)
        self.padding = kernel_size // 2
        self.weight = nn.Parameter(torch.randn(1, out_channel, in_channel, kernel_size, kernel_size))
        self.modulation = EqualLinear(style_dim, in_channel, bias_init=1)
        self.demodulate = demodulate

    def forward(self, input, style):
        batch, in_channel, height, width = input.shape

        style = self.modulation(style).view(batch, 1, in_channel, 1, 1)
        weight = self.scale * self.weight * style

        if self.demodulate:
            demod = torch.rsqrt(weight.pow(2).sum([2, 3, 4]) + 1e-8)
            weight = weight * demod.view(batch, self.out_channel, 1, 1, 1)

        weight = weight.view(batch * self.out_channel, in_channel, self.kernel_size, self.kernel_size)

        if self.upsample:
            input = input.view(1, batch * in_channel, height, width)
            weight = weight.view(batch, self.out_channel, in_channel, self.kernel_size, self.kernel_size)
            weight = weight.transpose(1, 2).reshape(
                batch * in_channel, self.out_channel, self.kernel_size, self.kernel_size)
            out = F.conv_transpose2d(input, weight, padding=0, stride=2, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)
            out = self.blur(out)
        else:
            input = input.view(1, batch * in_channel, height, width)
            out = F.conv2d(input, weight, padding=self.padding, groups=batch)
            _, _, height, width = out.shape
            out = out.view(batch, self.out_channel, height, width)

        return out


class NoiseInjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.zeros(1))

    def forward(self, image, noise=None):
        if noise is None:
            batch, _, height, width = image.shape
            noise = image.new_empty(batch, 1, height, width).normal_()
        return image + self.weight * noise


class ConstantInput(nn.Module):
    def __init__(self, channel, size=4):
        super().__init__()
        self.input = nn.Parameter(torch.randn(1, channel, size, size))

    def forward(self, input):
        return self.input.repeat(input.shape[0], 1, 1, 1)


class StyledConv(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size, style_dim, upsample=False,
                 blur_kernel=(1, 3, 3, 1), demodulate=True):
        super().__init__()
        self.conv = ModulatedConv2d(in_channel, out_channel, kernel_size, style_dim,
                                    upsample=upsample, blur_kernel=blur_kernel, demodulate=demodulate)
        self.noise = NoiseInjection()
        self.activate = FusedLeakyReLU(out_channel)

    def forward(self, input, style, noise=None):
        out = self.conv(input, style)
        out = self.noise(out, noise=noise)
        return self.activate(out)


class ToRGB(nn.Module):
    def __init__(self, in_channel, style_dim, upsample=True, blur_kernel=(1, 3, 3, 1)):
        super().__init__()
        if upsample:
            self.upsample = Upsample(blur_kernel)
        self.conv = ModulatedConv2d(in_channel, 3, 1, style_dim, demodulate=False)
        self.bias = nn.Parameter(torch.zeros(1, 3, 1, 1))

    def forward(self, input, style, skip=None):
        out = self.conv(input, style) + self.bias
        if skip is not None:
            out = out + self.upsample(skip)
        return out


class Generator(nn.Module):
    def __init__(self, size=1024, style_dim=512, n_mlp=8, channel_multiplier=2,
                 blur_kernel=(1, 3, 3, 1), lr_mlp=0.01):
        super().__init__()
        self.size = size
        self.style_dim = style_dim

        layers = [PixelNorm()]
        for _ in range(n_mlp):
            layers.append(EqualLinear(style_dim, style_dim, lr_mul=lr_mlp, activation='fused_lrelu'))
        self.style = nn.Sequential(*layers)

        self.channels = {
            4: 512, 8: 512, 16: 512, 32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        self.input = ConstantInput(self.channels[4])
        self.conv1 = StyledConv(self.channels[4], self.channels[4], 3, style_dim, blur_kernel=blur_kernel)
        self.to_rgb1 = ToRGB(self.channels[4], style_dim, upsample=False)

        self.log_size = int(math.log(size, 2))
        self.num_layers = (self.log_size - 2) * 2 + 1

        self.convs = nn.ModuleList()
        self.to_rgbs = nn.ModuleList()
        self.noises = nn.Module()

        for layer_idx in range(self.num_layers):
            res = (layer_idx + 5) // 2
            self.noises.register_buffer(f'noise_{layer_idx}', torch.randn(1, 1, 2 ** res, 2 ** res))

        in_channel = self.channels[4]
        for i in range(3, self.log_size + 1):
            out_channel = self.channels[2 ** i]
            self.convs.append(StyledConv(in_channel, out_channel, 3, style_dim,
                                         upsample=True, blur_kernel=blur_kernel))
            self.convs.append(StyledConv(out_channel, out_channel, 3, style_dim, blur_kernel=blur_kernel))
            self.to_rgbs.append(ToRGB(out_channel, style_dim))
            in_channel = out_channel

        self.n_latent = self.log_size * 2 - 2

    def forward(self, latent, randomize_noise=False):
        """Synthesize from a W+ latent of shape (batch, n_latent, style_dim)."""
        if randomize_noise:
            noise = [None] * self.num_layers
        else:
            noise = [getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)]

        out = self.input(latent)
        out = self.conv1(out, latent[:, 0], noise=noise[0])
        skip = self.to_rgb1(out, latent[:, 1])

        i = 1
        for conv1, conv2, noise1, noise2, to_rgb in zip(
                self.convs[::2], self.convs[1::2], noise[1::2], noise[2::2], self.to_rgbs):
            out = conv1(out, latent[:, i], noise=noise1)
            out = conv2(out, latent[:, i + 1], noise=noise2)
            skip = to_rgb(out, latent[:, i + 2], skip)
            i += 2

        return skip
