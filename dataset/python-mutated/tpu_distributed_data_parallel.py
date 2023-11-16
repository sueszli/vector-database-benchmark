import torch
from torch import nn
from fairseq.distributed import utils

class TPUDistributedDataParallel(nn.Module):

    def __init__(self, module, process_group):
        if False:
            return 10
        super().__init__()
        self.module = module
        self.process_group = process_group
        self.world_size = utils.get_world_size(self.process_group)

    def forward(self, *inputs, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        return self.module(*inputs, **kwargs)

    def all_reduce_grads(self):
        if False:
            return 10
        gradients = []
        for p in self.parameters():
            if not p.requires_grad:
                continue
            if p.grad is None:
                p.grad = torch.zeros_like(p)
            if p.grad.requires_grad:
                raise RuntimeError("TPUDistributedDataParallel only works with gradients that don't require grad")
            gradients.append(p.grad)
        import torch_xla.core.xla_model as xm
        xm.all_reduce('sum', gradients, scale=1.0 / self.world_size, groups=self.process_group[1])