import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from einops import rearrange
import numpy as np
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import multiprocessing as mp


# Self-Attention
class SelfAttn(nn.Module):
    def __init__(self, dim, num_heads=8, bias=False):
        super(SelfAttn, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.eps = 1e-6

        self.qkv = nn.Linear(dim, dim * 3, bias=bias)
        self.proj_out = nn.Linear(dim, dim)

    def softplus_feature_map(self, x):
        return torch.nn.functional.softplus(x)

    def forward(self, x):
        qkv = self.qkv(x).chunk(3, dim=-1)
        # [b, N, c] -> [b, N, head, c//head] -> [b, head, N, c//head]
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.num_heads), qkv)

        # [b, head, N, c//head] * [b, head, N, c//head] -> [b, head, N, N]
        # attn = torch.einsum('bijc, bikc -> bijk', q, k) * self.scale
        # attn = attn.softmax(dim=-1)
        # # [b, head, N, N] * [b, head, N, c//head] -> [b, head, N, c//head] -> [b, N, head, c//head]
        # x = torch.einsum('bijk, bikc -> bijc', attn, v)
        # x = rearrange(x, 'b i j c -> b j (i c)')

        q = self.softplus_feature_map(F.normalize(q, dim=-1))
        k = self.softplus_feature_map(F.normalize(k, dim=-1)).permute(0, 1, 3, 2)

        kv = torch.einsum("bhmn, bhnc->bhmc", k, v) * self.scale

        norm = 1 / torch.einsum("bhnc, bhc->bhn", q, torch.sum(k, dim=-1) + self.eps)
        x = torch.einsum("bhnm, bhmc, bhn->bhnc", q, kv, norm)
        x = rearrange(x, 'b i j c -> b j (i c)')

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, mlp_ratio=4):
        super(Mlp, self).__init__()
        hidden_features = in_features * mlp_ratio

        self.fc = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            # nn.GELU(),
            nn.Linear(hidden_features, in_features)
        )

    def forward(self, x):
        return self.fc(x)


def window_partition(x, window_size):
    """
    Args:
        x: (b, h, w, c)
        window_size (int): window size
    Returns:
        windows: (num_windows*b, window_size, window_size, c) [non-overlap]
    """
    return rearrange(x, 'b (h s1) (w s2) c -> (b h w) s1 s2 c', s1=window_size, s2=window_size)


def window_reverse(windows, window_size, h, w):
    """
    Args:
        windows: (num_windows*B, window_size, window_size, C)
        window_size (int): Window size
        h (int): Height of image
        w (int): Width of image
    Returns:
        x: (b, h, w, c)
    """
    b = int(windows.shape[0] / (h * w / window_size / window_size))
    return rearrange(windows, '(b h w) s1 s2 c -> b (h s1) (w s2) c', b=b, h=h // window_size, w=w // window_size)


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.relu1 = nn.ReLU6()

        self.conv2 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.relu2 = nn.ReLU6()

    def forward(self, x):
        x = self.conv1(x)
        # x = self.relu1(x)

        x = self.conv2(x)
        # x = self.relu2(x)
        return x


class Transformer(nn.Module):
    def __init__(self, dim, num_heads=7, window_size=15, mlp_ratio=4, qkv_bias=True): # window_size=6
        super(Transformer, self).__init__()
        self.window_size = window_size
        self.pos_embed = nn.Conv2d(dim, dim, 3, padding=1, groups=dim)

        self.attn = SelfAttn(dim, num_heads, qkv_bias)

        self.mlp = Mlp(dim, mlp_ratio)

    def forward(self, x):
        x = x + self.pos_embed(x)
        x = rearrange(x, 'b c h w -> b h w c')
        b, h, w, c = x.shape

        shortcut = x

        pad_l = pad_t = 0
        pad_r = (self.window_size - w % self.window_size) % self.window_size
        pad_b = (self.window_size - h % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, pad_l, pad_r, pad_t, pad_b))
        _, Hp, Wp, _ = x.shape

        x_windows = window_partition(x, self.window_size)  # nW*B, window_size, window_size, c
        x_windows = rearrange(x_windows, 'B s1 s2 c -> B (s1 s2) c', s1=self.window_size,
                              s2=self.window_size)  # nW*b, window_size*window_size, c

        # W-MSA/SW-MSA
        attn_windows = self.attn(x_windows)  # nW*b, window_size*window_size, c

        # merge windows
        attn_windows = rearrange(attn_windows, 'B (s1 s2) c -> B s1 s2 c', s1=self.window_size, s2=self.window_size)
        x = window_reverse(attn_windows, self.window_size, Hp, Wp)  # b H' W' c

        # reverse cyclic shift
        if pad_r > 0 or pad_b > 0:
            x = x[:, :h, :w, :].contiguous()

        x = x + shortcut
        x = x + self.mlp(x)
        return rearrange(x, 'b h w c -> b c h w')


class ResBlock(nn.Module):
    def __init__(self, in_features, ratio=2):
        super(ResBlock, self).__init__()

        self.down_conv = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(in_features, in_features * ratio, kernel_size=7, stride=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features * ratio, in_features * ratio, 3, 1, ),
            nn.ReLU6()
        )

        self.net = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features * ratio, in_features * ratio, 3, 1, ),
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_features * ratio, in_features * ratio, 3, 1, ),
            nn.ReLU6(),
            nn.Conv2d(in_features * ratio, in_features * ratio, 1, 1, 0),
        )

    def forward(self, x):
        x = self.down_conv(x)
        return self.net(x) + x


class BaseBlock(nn.Module):
    def __init__(self, num_heads=8, window_size=5, channels=[64, 128, 256, 512, 1024], qkv_bias=True, ratio=1):
        super(BaseBlock, self).__init__()
        self.layers = nn.ModuleList([])
        for channel in channels:
            self.layers.append(nn.ModuleList([
                ResBlock(channel, ratio),
                Transformer(channel * ratio, num_heads, window_size, qkv_bias),
            ]))

    def forward(self, x):
        features = []
        for rblock, tblock in self.layers:
            x = rblock(x)
            x = tblock(x)
        return x

class FlowFormer(nn.Module):
    def __init__(self, num_classes, n_heads=8, channels=[32, 32, 32], ratio=1, resolution=None, control_number=None, lookback=5):
        super(FlowFormer, self).__init__()
        self.name = 'FlowFormer'

        if resolution is None:
            resolution = [570, 840]
        if control_number is None:
            control_number = resolution[0] + 24

        self.resolution = resolution
        self.look_back = lookback
        self.target_feature = resolution[0] * resolution[1]

        self.linear = nn.Sequential(
            nn.Linear(control_number, resolution[0] * resolution[1]))

        self.input = nn.Conv2d(lookback, channels[0], (1, 1), (1, 1), (0, 0))

        self.conv_head = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(lookback, channels[0], kernel_size=3, stride=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1),
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1),
        )

        self.body = BaseBlock(num_heads=n_heads, channels=channels, ratio=ratio)

        self.decoder = Decoder(channels)

        self.final_control = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1),
            nn.Conv2d(channels[0], num_classes, kernel_size=1)
        )

        self.final = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(channels[0], channels[0], kernel_size=3, stride=1),
            nn.Conv2d(channels[0], num_classes, kernel_size=1)
        )

    def forward(self, x, control):
        b, c, h, w = x.size()

        x = self.conv_head(x)
        x = self.final(x)

        control = self.linear(control).reshape(b, -1, h, w)
        control = self.input(control)
        control = self.body(control)
        control = self.final_control(control)

        if self.training:
            return (x + control).squeeze(1), control.squeeze(1)

        return (x + control).squeeze(1)


if __name__ == '__main__':
    a = torch.randn(4, 5, 126, 210).cuda()
    control = torch.randn(4, 5, 127).cuda()
    model = FlowFormer(1).cuda()
    output = model(a, control)
    print(output.shape)
