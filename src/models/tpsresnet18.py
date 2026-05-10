import torch
import torch.nn as nn
import torchvision.models as models


class TPSResNet18(nn.Module):
    def __init__(self, num_points: int = 81, use_coordconv: bool = False):
        super().__init__()

        self.use_coordconv = use_coordconv
        in_channels = 3 if use_coordconv else 1

        # ===== Backbone =====
        self.backbone = models.resnet18(weights=None)

        # Меняем первый слой под 1 канал (или 3 если CoordConv)
        self.backbone.conv1 = nn.Conv2d(
            in_channels,
            64,
            kernel_size=7,
            stride=2,
            padding=3,
            bias=False
        )

        # Убираем классификатор
        self.backbone.fc = nn.Identity()

        # ===== Regression head =====
        self.regressor = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_points * 2)
        )

    def forward(self, x):

        # ===== CoordConv (опционально) =====
        if self.use_coordconv:
            B, _, H, W = x.shape

            yy, xx = torch.meshgrid(
                torch.linspace(-1, 1, H, device=x.device),
                torch.linspace(-1, 1, W, device=x.device),
                indexing="ij"
            )

            xx = xx.expand(B, 1, H, W)
            yy = yy.expand(B, 1, H, W)

            x = torch.cat([x, xx, yy], dim=1)  # (B, 3, H, W)

        # ===== Backbone =====
        features = self.backbone(x)  # (B, 512)

        # ===== Regression =====
        delta = self.regressor(features)  # (B, num_points * 2)

        return delta
