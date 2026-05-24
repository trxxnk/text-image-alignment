import torch
import torch.nn as nn
import torchvision.models as models


def _adapt_conv1_from_rgb(
    old_weight: torch.Tensor,
    in_channels: int,
) -> torch.Tensor:
    """(64,3,7,7) -> (64,in_channels,7,7) для grayscale или CoordConv."""
    out_ch, _, kh, kw = old_weight.shape
    new_w = torch.zeros(out_ch, in_channels, kh, kw, dtype=old_weight.dtype, device=old_weight.device)
    rgb_mean = old_weight[:, :3].mean(dim=1)  # (64,7,7)
    new_w[:, 0] = rgb_mean
    if in_channels > 1:
        nn.init.kaiming_normal_(new_w[:, 1:], mode="fan_out", nonlinearity="relu")
        new_w[:, 1:].mul_(0.01)
    return new_w


class TPSResNet18(nn.Module):
    def __init__(
        self,
        num_points: int = 81,
        use_coordconv: bool = False,
        pretrained: bool = False,
        output_scale: float = 0.25,
    ):
        super().__init__()

        self.use_coordconv = use_coordconv
        in_channels = 3 if use_coordconv else 1
        self.output_scale = float(output_scale)

        if pretrained:
            try:
                from torchvision.models import ResNet18_Weights

                self.backbone = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
            except Exception:  # pragma: no cover - старые torchvision
                self.backbone = models.resnet18(pretrained=True)
        else:
            self.backbone = models.resnet18(weights=None)

        old_conv = self.backbone.conv1
        new_conv = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False,
        )
        if pretrained:
            new_conv.weight.data.copy_(_adapt_conv1_from_rgb(old_conv.weight.data, in_channels))
        self.backbone.conv1 = new_conv

        self.backbone.fc = nn.Identity()

        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_points * 2),
        )

    def forward(self, x):
        if self.use_coordconv:
            B, _, H, W = x.shape

            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing="ij",
            )

            xx = xx.expand(B, 1, H, W)
            yy = yy.expand(B, 1, H, W)

            x = torch.cat([x, xx, yy], dim=1)

        features = self.backbone(x)
        delta = self.regressor(features)
        if self.output_scale > 0:
            delta = torch.tanh(delta) * self.output_scale
        return delta
