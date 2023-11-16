import torch.nn.functional as F
import torch
import numpy as np

def cross_loss(output, target):
    if False:
        for i in range(10):
            print('nop')
    target = target.cpu().repeat_interleave(15, 0).repeat_interleave(8, 1).cuda().long()
    assert output.shape[0] == target.shape[0]
    return F.cross_entropy(output, target)