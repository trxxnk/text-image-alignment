import torch.nn as nn
import torch.nn.functional as F


class TPSLossRegularization(nn.Module):
    def __init__(self, grid_size: int = 9, lambda_smooth: float = 0.05):
        super().__init__()
        self.grid_size = int(grid_size)
        self.lambda_smooth = float(lambda_smooth)

    def forward(self, pred, target):
        b = pred.size(0)
        pred = pred.view(b, self.grid_size, self.grid_size, 2)
        target = target.view(b, self.grid_size, self.grid_size, 2)
        data_loss = F.smooth_l1_loss(pred, target)
        dx = pred[:, :, 1:, :] - pred[:, :, :-1, :]
        dy = pred[:, 1:, :, :] - pred[:, :-1, :, :]
        smooth_loss = dx.norm(dim=-1).mean() + dy.norm(dim=-1).mean()
        return data_loss + self.lambda_smooth * smooth_loss
