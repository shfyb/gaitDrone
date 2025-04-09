import torch.nn.functional as F
import torch
from .base import BaseLoss
import pdb

class CrossEntropyLoss_uncertainty(BaseLoss):
    def __init__(self, scale=2**4, label_smooth=True, eps=0.1, loss_term_weight=1.0, log_accuracy=False):
        super(CrossEntropyLoss_uncertainty, self).__init__(loss_term_weight)
        self.scale = scale
        self.label_smooth = label_smooth
        self.eps = eps
        self.log_accuracy = log_accuracy

    def forward(self, logits, var, labels):
        """
            logits: [n, c, p]
            labels: [n]
        """
        n, c, p = logits.size()
        logits = logits.float()
        
        labels = labels.unsqueeze(1).repeat(1,p)
        
        total_loss = 0.0
        for part in range(p):
            
            part_logits = logits[:, :, part]  # 当前部分的logits
            part_labels = labels[:, part]  # 当前部分的标签
            part_var = var[:, :, part]  # 当前部分的方差

            # 计算当前部分的交叉熵损失
            ce_loss = F.cross_entropy(part_logits, part_labels, label_smoothing=self.eps)

            # 调整损失
            adjusted_loss = (ce_loss / (part_var.squeeze()**2).mean())

            log_var_term = torch.log((part_var.squeeze()**2).mean())

            # 求和或平均，取决于你的具体需求
            adjusted_loss = adjusted_loss+log_var_term  # 举例使用平均

            # 累加到总损失
            total_loss += adjusted_loss

            # 可以选择对总损失进行平均或其他处理
        loss = total_loss / p
        # if self.label_smooth:
        #     pdb.set_trace()
        #     loss = F.cross_entropy(
        #         logits*self.scale, labels.repeat(1, p), label_smoothing=self.eps)
            
        # else:
        #     loss = F.cross_entropy(logits*self.scale, labels.repeat(1, p))
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = logits.argmax(dim=1)  # [n, p]
            accu = (pred == labels).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info
