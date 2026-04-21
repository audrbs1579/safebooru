import torch
import torch.nn as nn


class AsymmetricLoss(nn.Module):
    """멀티레이블 불균형 완화용 손실 함수."""
    def __init__(self, gamma_neg: float = 4, gamma_pos: float = 0, clip: float = 0.05):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.clip = clip

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        xs_pos = torch.sigmoid(logits)
        xs_neg = (1 - xs_pos + self.clip).clamp(max=1)
        loss = (
            targets * torch.log(xs_pos.clamp(min=1e-8)) * (1 - xs_pos).pow(self.gamma_pos) +
            (1 - targets) * torch.log(xs_neg.clamp(min=1e-8)) * xs_neg.pow(self.gamma_neg)
        )
        return -loss.mean()
