import torch
import torch.nn as nn
import torch.nn.functional as F


class TPSLossRegularization(nn.Module):
    def __init__(self, grid_size: int = 9, lambda_smooth: float = 0.05):
        super().__init__()
        self.grid_size = int(grid_size)
        self.lambda_smooth = float(lambda_smooth)

    def per_sample_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Скаляр лосса на каждый элемент батча, shape (B,)."""
        b = pred.size(0)
        pred = pred.view(b, self.grid_size, self.grid_size, 2)
        target = target.view(b, self.grid_size, self.grid_size, 2)
        per_el = F.smooth_l1_loss(pred, target, reduction="none")
        data = per_el.mean(dim=(1, 2, 3))
        dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        smooth_x = dx.norm(dim=-1).mean(dim=(1, 2))
        smooth_y = dy.norm(dim=-1).mean(dim=(1, 2))
        smooth = smooth_x + smooth_y
        return data + self.lambda_smooth * smooth

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.per_sample_loss(pred, target).mean()
