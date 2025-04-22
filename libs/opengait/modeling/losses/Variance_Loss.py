import torch.nn.functional as F
import torch
from .base import BaseLoss
import pdb

class VarianceLoss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, gamma=1e5, log_accuracy=False):
        super(VarianceLoss, self).__init__(loss_term_weight)
        self.gamma = gamma
        self.log_accuracy = log_accuracy

    def forward(self, variance):
        """
            logits: [n, c, p]
            labels: [n]
        """
        zero_tensor = torch.zeros_like(variance)

        # 计算 loss
        loss = torch.max(zero_tensor, self.gamma - variance)


        self.info.update({'loss': loss.detach().clone()})

        return loss, self.info

