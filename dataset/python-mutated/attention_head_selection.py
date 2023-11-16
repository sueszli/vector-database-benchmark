import math
import torch
from torch.nn.modules.loss import _Loss

class HeadSelectionLoss(_Loss):

    def __init__(self, args):
        if False:
            while True:
                i = 10
        super().__init__()
        self.args = args
        self.kl_weight = getattr(args, 'kl_weight', 0.0)

    def forward(self, head_samples, sample_sizes, prior=0.5, eps=1e-07):
        if False:
            for i in range(10):
                print('nop')
        '\n        head_scores: (num_tasks, num_layers, num_heads)\n        sample_sizes: (num_tasks, )\n        '
        kl_loss = (head_samples * (torch.log(head_samples + eps) - math.log(prior))).sum(-1).sum(-1)
        kl_loss /= torch.numel(head_samples) / head_samples.size(0)
        kl_loss = self.kl_weight * torch.matmul(kl_loss, sample_sizes)
        return kl_loss