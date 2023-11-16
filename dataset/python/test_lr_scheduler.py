# -*- coding: utf-8 -*-
from bisect import bisect_right

import numpy as np

from megengine import Parameter, tensor
from megengine.module import Module
from megengine.optimizer import SGD, MultiStepLR


class Simple(Module):
    def __init__(self):
        super().__init__()
        self.a = Parameter([1.23], dtype=np.float32)

    def forward(self, x):
        x = x * self.a
        return x


def test_multi_step_lr():
    net = Simple()
    opt = SGD(net.parameters(), lr=0.01, momentum=0.9)
    scheduler = MultiStepLR(opt, [3, 6, 8])

    lr = np.array(0.01, dtype=np.float32)
    for i in range(10):
        for group in opt.param_groups:
            np.testing.assert_almost_equal(
                np.array(group["lr"], dtype=np.float32),
                (lr * 0.1 ** bisect_right([3, 6, 8], i)).astype(np.float32),
            )
        scheduler.step()
