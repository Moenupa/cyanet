import torch
from torch import nn
from torch.nn import Module, Parameter
from .mhca import MultiHeadChannelAttention


class ColorFusion(Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.conv_y = nn.Conv2d(c, c, kernel_size=1)
        self.conv_uv = nn.Conv2d(c, c, kernel_size=1)
        self.alpha = Parameter(
            torch.zeros(1, 1, 1, 1),
            requires_grad=True
        )
        self.ca = SimplifiedChannelAttention(c)

    def forward(self, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor):
        # y: (B, c, H, W); u, v: (B, c // 2, H, W)
        uv = torch.cat([u, v], dim=1)
        excited_uv = self.alpha * self.conv_y(y) + self.conv_uv(uv)
        uv = self.ca(excited_uv) + uv + excited_uv
        return torch.cat([y, uv], dim=1)


class SimplifiedChannelAttention(Module):
    def __init__(self, c: int) -> None:
        super().__init__()
        self.lnorm = nn.GroupNorm(1, c)
        self.dwconv = nn.Conv2d(c, c, kernel_size=3,
                                stride=1, padding=1, groups=c)
        self.ca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(c, c, kernel_size=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        x = self.lnorm(x)
        return x * self.dwconv(x) * self.ca(x)


class Denoiser(Module):
    """
    Color Denoiser

    Args:
        c (int): channels in hidden dimension
    """

    def __init__(self, c: int) -> None:
        super().__init__()
        # U net architecture
        # (B, 1, H, W) -> (B, c, H/2, W/2) -> (B, 1, H, W)

        self.intro = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.attn = MultiHeadChannelAttention(c, c, num_heads=4)

        self.up4 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up3 = nn.UpsamplingBilinear2d(scale_factor=2)
        self.up2 = nn.UpsamplingBilinear2d(scale_factor=2)

        self.out1 = nn.Sequential(
            nn.Conv2d(c, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )
        self.out0 = nn.Sequential(
            nn.Conv2d(1, 1, kernel_size=1, stride=1),
            nn.Tanh()
        )

    def forward(self, x0: torch.Tensor):
        """
        Args:
            x (torch.Tensor): (B, 1, H, W)
        """
        x1 = self.intro(x0)  # (B, c, H, W)
        x2 = self.conv1(x1)  # (B, c, H/2, W/2)
        x3 = self.conv2(x2)  # (B, c, H/4, W/4)
        x4 = self.conv3(x3)  # (B, c, H/8, W/8)
        x4 = self.attn(x4)

        u3 = self.up4(x4)
        u2 = self.up3(u3 + x3)
        u1 = self.up2(u2 + x2)

        u0 = self.out1(u1 + x1)
        out = self.out0(u0 + x0)
        return out


class Cyanet(Module):
    """Cyanet: A Channel-wise YUV-Abstracted Net for low-light image enhancement

    Args:
    """

    def __init__(self, c: int = 32) -> None:
        super().__init__()
        assert c % 2 == 0, 'c must be even'

        self.u_denoiser = Denoiser(c // 2)
        self.v_denoiser = Denoiser(c // 2)

        self.y_intro = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.y_attn = nn.Sequential(
            nn.MaxPool2d(8),
            MultiHeadChannelAttention(c, c, num_heads=4),
            nn.UpsamplingBilinear2d(scale_factor=8)
        )

        self.u_conv = nn.Sequential(
            nn.Conv2d(1, c // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.v_conv = nn.Sequential(
            nn.Conv2d(1, c // 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )

        self.fusion = ColorFusion(c)

        self.final_conv = nn.Sequential(
            nn.Conv2d(2*c, c, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(c, 3, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x: torch.Tensor):
        # x: (B, 3, H, W), split to y, u, v (B, 1, H, W)
        y, u, v = x.split(1, dim=1)
        u = u + self.u_denoiser(u)
        v = v + self.v_denoiser(v)

        y = self.y_intro(y)
        y = y + self.y_attn(y)
        u = self.u_conv(u)
        v = self.v_conv(v)

        yuv = self.fusion(y, u, v)
        y, u, v = self.final_conv(yuv).split(1, dim=1)
        yuv_pred = torch.cat([(y + 1.) / 2., u, v], dim=1)
        return yuv_pred


def frange(x: torch.Tensor):
    if 3 in x.shape:
        c1, c2, c3 = x.split(1, dim=x.shape.index(3))
        return (
            'c1', c1.min(), c1.max(),
            'c2', c2.min(), c2.max(),
            'c3', c3.min(), c3.max()
        )
    return (x.min(), x.max())
