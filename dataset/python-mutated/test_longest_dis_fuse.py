import sys
import os
import jittor as jt
import unittest
import time
import numpy as np

def get_init_var(shape, dtype):
    if False:
        i = 10
        return i + 15
    return jt.random(shape, dtype)

def pool(x, size, op, padding, stride=1):
    if False:
        while True:
            i = 10
    (N, C, H, W) = x.shape
    h = (H + padding * 2 - size) // stride + 1
    w = (W + padding * 2 - size) // stride + 1
    xx = x.reindex([N, C, h, w, size, size], ['i0', 'i1', f'i2*{stride}-{padding}+i4', f'i3*{stride}-{padding}+i5'])
    return xx.reindex_reduce(op, [N, C, h, w], ['i0', 'i1', 'i2', 'i3'])

def relu(x):
    if False:
        while True:
            i = 10
    return jt.maximum(x, jt.float32(0))

def resnet_fake():
    if False:
        while True:
            i = 10
    from jittor import nn
    net = nn.Sequential(nn.Conv(3, 64, 7, 2, 3), nn.BatchNorm(64), nn.ReLU(), nn.Pool(3, 2, 1))
    return net

class TestLongestDisFuse(unittest.TestCase):

    def test_longest_dis_fuse(self):
        if False:
            i = 10
            return i + 15
        x = jt.array(np.random.rand(1, 3, 224, 224).astype(np.float32))
        net = resnet_fake()
        loss = jt.sum(net(x))
        ps = net.parameters()
        gs = jt.grad(loss, ps)
        jt.sync(gs)
        g = jt.dump_all_graphs()
        for s in g.nodes_info:
            if not s.startswith('Var'):
                continue
            shape = s.split('[')[1].split(']')[0].split(',')
            ptr = s.split('(')[1].split(')')[0].split(',')[-1]
            if ptr != '0' and ptr != '0x0':
                assert len(shape) <= 5, s
if __name__ == '__main__':
    unittest.main()