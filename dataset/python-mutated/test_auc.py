import math
import torch
from catalyst.metrics.functional._auc import auc

def test_auc():
    if False:
        for i in range(10):
            print('nop')
    '\n    Tests for catalyst.metrics.auc metric.\n    '
    test_size = 1000
    scores = torch.cat((torch.rand(test_size), torch.rand(test_size)))
    targets = torch.cat((torch.zeros(test_size), torch.ones(test_size)))
    val = auc(scores, targets)
    assert math.fabs(val - 0.5) < 0.1, 'AUC test1 failed'
    scores = torch.cat((torch.Tensor(test_size).fill_(0), torch.Tensor(test_size).fill_(0.1), torch.Tensor(test_size).fill_(0.2), torch.Tensor(test_size).fill_(0.3), torch.Tensor(test_size).fill_(0.4), torch.ones(test_size)))
    targets = torch.cat((torch.zeros(test_size), torch.zeros(test_size), torch.zeros(test_size), torch.zeros(test_size), torch.zeros(test_size), torch.ones(test_size)))
    val = auc(scores, targets)
    assert math.fabs(val - 1.0) < 0.0001, 'AUC test2 failed'