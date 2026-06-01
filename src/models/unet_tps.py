from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as tv_models


def _add_coord_channels(x: torch.Tensor) -> torch.Tensor:
    """К (B,1,H,W) добавляет 2 канала нормированных координат -> (B,3,H,W)."""
    b, _, h, w = x.shape
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, h, device=x.device),
        torch.linspace(-1, 1, w, device=x.device),
        indexing="ij",
    )
    xx = xx.expand(b, 1, h, w)
    yy = yy.expand(b, 1, h, w)
    return torch.cat([x, xx, yy], dim=1)


def _sample_field_on_grid(field: torch.Tensor, grid_size: int, output_scale: float) -> torch.Tensor:
    """field (B,2,h,w) -> (B, grid_size*grid_size*2).

    Сэмплируем плотное поле в узлах равномерной сетки grid_size x grid_size
    (align_corners=True -> крайние узлы попадают в углы, как базовая сетка
    linspace(0, W-1, N)). Порядок: y внешний, x внутренний; канал = (dx, dy).
    """
    b = field.shape[0]
    lin = torch.linspace(-1.0, 1.0, grid_size, device=field.device, dtype=field.dtype)
    yy, xx = torch.meshgrid(lin, lin, indexing="ij")  # строки=y, столбцы=x
    sample_grid = torch.stack([xx, yy], dim=-1).unsqueeze(0).expand(b, -1, -1, -1)
    sampled = F.grid_sample(field, sample_grid, align_corners=True)  # (B,2,N,N)
    out = sampled.permute(0, 2, 3, 1).reshape(b, grid_size * grid_size * 2)
    if output_scale > 0:
        out = torch.tanh(out) * output_scale
    return out


class _DoubleConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class _Up(nn.Module):
    """Upsample(bilinear) + конкатенация скипа + DoubleConv."""

    def __init__(self, in_ch: int, skip_ch: int, out_ch: int):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
        self.conv = _DoubleConv(in_ch + skip_ch, out_ch)

    def forward(self, x, skip):
        x = self.up(x)
        # подгонка размеров на случай нечётностей
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        x = torch.cat([x, skip], dim=1)
        return self.conv(x)


class UNetTPS(nn.Module):
    """Классический U-Net; декодер -> поле (B,2,H,W) -> сэмпл в сетке N x N."""

    def __init__(
        self,
        num_points: int = 81,
        use_coordconv: bool = False,
        pretrained: bool = False,  # не используется; для совместимости с фабрикой
        output_scale: float = 0.25,
        base_ch: int = 32,
    ):
        super().__init__()
        self.use_coordconv = use_coordconv
        self.output_scale = float(output_scale)
        self.grid_size = int(round(math.sqrt(num_points)))
        if self.grid_size * self.grid_size != num_points:
            raise ValueError(f"num_points={num_points} не является полным квадратом")

        in_ch = 3 if use_coordconv else 1
        c1, c2, c3, c4 = base_ch, base_ch * 2, base_ch * 4, base_ch * 8

        self.enc1 = _DoubleConv(in_ch, c1)
        self.enc2 = _DoubleConv(c1, c2)
        self.enc3 = _DoubleConv(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.bottleneck = _DoubleConv(c3, c4)

        self.up3 = _Up(c4, c3, c3)
        self.up2 = _Up(c3, c2, c2)
        self.up1 = _Up(c2, c1, c1)
        self.head = nn.Conv2d(c1, 2, kernel_size=1)

    def forward(self, x):
        if self.use_coordconv:
            x = _add_coord_channels(x)

        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        b = self.bottleneck(self.pool(e3))

        d3 = self.up3(b, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        field = self.head(d1)  # (B,2,H,W)

        return _sample_field_on_grid(field, self.grid_size, self.output_scale)


def _adapt_conv1_from_rgb(old_weight: torch.Tensor, in_channels: int) -> torch.Tensor:
    out_ch, _, kh, kw = old_weight.shape
    new_w = torch.zeros(out_ch, in_channels, kh, kw, dtype=old_weight.dtype, device=old_weight.device)
    new_w[:, 0] = old_weight[:, :3].mean(dim=1)
    if in_channels > 1:
        nn.init.kaiming_normal_(new_w[:, 1:], mode="fan_out", nonlinearity="relu")
        new_w[:, 1:].mul_(0.01)
    return new_w


class ResUNetTPS(nn.Module):
    """Энкодер ResNet18 (опц. предобученный) + U-Net-декодер -> поле -> сэмпл N x N."""

    def __init__(
        self,
        num_points: int = 81,
        use_coordconv: bool = False,
        pretrained: bool = False,
        output_scale: float = 0.25,
    ):
        super().__init__()
        self.use_coordconv = use_coordconv
        self.output_scale = float(output_scale)
        self.grid_size = int(round(math.sqrt(num_points)))
        if self.grid_size * self.grid_size != num_points:
            raise ValueError(f"num_points={num_points} не является полным квадратом")

        in_ch = 3 if use_coordconv else 1

        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights

                backbone = tv_models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:  # pragma: no cover
                backbone = tv_models.resnet18(pretrained=True)
        else:
            backbone = tv_models.resnet18(weights=None)

        old_conv = backbone.conv1
        new_conv = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
        if pretrained:
            new_conv.weight.data.copy_(_adapt_conv1_from_rgb(old_conv.weight.data, in_ch))
        backbone.conv1 = new_conv

        # ступени энкодера (каналы: 64, 64, 128, 256, 512)
        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)  # /2, 64
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1  # /4,  64
        self.layer2 = backbone.layer2  # /8,  128
        self.layer3 = backbone.layer3  # /16, 256
        self.layer4 = backbone.layer4  # /32, 512

        self.up3 = _Up(512, 256, 256)  # -> /16
        self.up2 = _Up(256, 128, 128)  # -> /8
        self.up1 = _Up(128, 64, 64)    # -> /4
        self.head = nn.Conv2d(64, 2, kernel_size=1)

    def forward(self, x):
        if self.use_coordconv:
            x = _add_coord_channels(x)

        s = self.stem(x)          # 64, /2
        x1 = self.layer1(self.maxpool(s))  # 64,  /4
        x2 = self.layer2(x1)      # 128, /8
        x3 = self.layer3(x2)      # 256, /16
        x4 = self.layer4(x3)      # 512, /32

        d3 = self.up3(x4, x3)     # 256, /16
        d2 = self.up2(d3, x2)     # 128, /8
        d1 = self.up1(d2, x1)     # 64,  /4
        field = self.head(d1)     # (B,2,H/4,W/4)

        return _sample_field_on_grid(field, self.grid_size, self.output_scale)
