import torch
from torch import nn
import torch.nn.functional as F


class SSIMLoss(nn.Module):

    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        B = X.size(0)

        # Flatten spatial dimensions, keep batch
        X = X.view(B, -1)
        Y = Y.view(B, -1)

        # Compute means
        X_mean = X.mean(dim=1, keepdim=True)
        Y_mean = Y.mean(dim=1, keepdim=True)

        # Centered versions
        Xc = X - X_mean
        Yc = Y - Y_mean

        # Covariance and variances per sample
        cov_xy = torch.mean(Xc * Yc, dim=1)
        var_x = torch.mean(Xc ** 2, dim=1)
        var_y = torch.mean(Yc ** 2, dim=1)

        # Compute loss per sample
        loss_per_sample = 1 - cov_xy / (var_x + var_y + self.eps)

        # Average over batch
        return loss_per_sample.mean()


class CombinedLoss(nn.Module):

    def __init__(
        self, 
        eps: float = 1e-6,
        alpha: float = 1.0,
        beta: float = 1.0,
        normalize: bool = False
    ) -> None:
        super().__init__()

        self.alpha = alpha
        self.beta = beta
        self.normalize = normalize

        self.l1_loss = nn.SmoothL1Loss()
        self.ssim_loss = SSIMLoss(eps=eps)

    def forward(self, X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
        denominator = self.alpha + self.beta if self.normalize else 1
        
        loss = (self.alpha * self.l1_loss(X, Y) + self.beta * self.ssim_loss(X, Y)) / denominator
        return loss
