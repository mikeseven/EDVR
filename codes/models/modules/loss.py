import torch
import torch.nn as nn


class CharbonnierLoss(nn.Module):
    """Charbonnier Loss (L1)"""

    def __init__(self, eps=1e-3, reduction='mean'):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps*eps
        self.reduction_fn = torch.mean if reduction == 'mean' else torch.sum

    def forward(self, x, y):
        diff = x - y
        loss = self.reduction_fn(torch.sqrt(diff * diff + self.eps))
        return loss
