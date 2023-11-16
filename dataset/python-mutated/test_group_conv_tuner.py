import unittest
import jittor as jt
import os
import numpy as np
from jittor import compile_extern
from jittor.test.test_log import find_log_with_re
if jt.has_cuda:
    from jittor.compile_extern import cublas_ops, cudnn_ops
else:
    cublas_ops = cudnn_ops = None

def conv_nchw(x, in_planes, out_planes, kernel_size, padding, stride=1, dilation=1, groups=1, init_method=None, w_=None):
    if False:
        for i in range(10):
            print('nop')
    (N, C, H, W) = x.shape
    (Kh, Kw) = (kernel_size, kernel_size)
    G = groups
    CpG = C // G
    padding = (padding, padding)
    dilation = (dilation, dilation)
    stride = (stride, stride)
    assert C == in_planes
    oc = out_planes
    oh = (H + padding[0] * 2 - Kh * dilation[0] + dilation[0] - 1) // stride[0] + 1
    ow = (W + padding[1] * 2 - Kw * dilation[1] + dilation[1] - 1) // stride[1] + 1
    if w_ is None:
        assert 0
    else:
        w = w_
    xx = x.reindex([N, G, oc // G, CpG, oh, ow, Kh, Kw], ['i0', f'i1*{CpG}+i3', f'i4*{stride[0]}-{padding[0]}+i6*{dilation[0]}', f'i5*{stride[1]}-{padding[1]}+i7*{dilation[1]}'])
    ww = w.reindex([N, G, oc // G, CpG, oh, ow, Kh, Kw], [f'i1*{oc // G}+i2', 'i3', 'i6', 'i7'])
    yy = xx * ww
    y = yy.reindex_reduce('add', [N, oc, oh, ow], ['i0', f'i1*{oc // G}+i2', 'i4', 'i5'])
    return y

def test_nchw(x, w, stride, padding, dilation, groups):
    if False:
        return 10
    (_, in_planes, _, _) = x.shape
    (out_planes, _, kernel_size, _) = w.shape
    return conv_nchw(x, in_planes, out_planes, kernel_size, padding, stride=stride, dilation=dilation, groups=groups, w_=w)

def check_forward(xshape, wshape, stride, padding, dilation, groups, use_cuda, nhwc):
    if False:
        print('Hello World!')
    assert nhwc == 0
    test_func = test_nchw
    with jt.log_capture_scope(use_cuda=use_cuda, enable_tuner=1, log_v=10, log_vprefix='op.cc=100,conv_tuner=1000') as raw_log:
        x = jt.random(xshape)
        w = jt.random(wshape)
        y = test_func(x, w, stride, padding, dilation, groups)
        y.sync()
    with jt.flag_scope(use_cuda=0, enable_tuner=0):
        cy = test_func(x, w, stride, padding, dilation, groups)
        cy.sync()
    logs = find_log_with_re(raw_log, '(Jit op key (not )?found: .*conv.*)')
    assert len(logs) == 1
    assert np.allclose(y.data, cy.data)

def check_backward(xshape, wshape, stride, padding, dilation, groups, use_cuda, nhwc):
    if False:
        i = 10
        return i + 15
    assert nhwc == 0
    test_func = test_nchw
    with jt.log_capture_scope(use_cuda=use_cuda, enable_tuner=1, log_v=10, log_vprefix='op.cc=100,conv_tuner=1000') as raw_log:
        x = jt.random(xshape)
        w = jt.random(wshape)
        y = test_func(x, w, stride, padding, dilation, groups)
        y.sync()
        (dx, dw) = jt.grad(y, [x, w])
        jt.sync([y, dx, dw])
    with jt.flag_scope(use_cuda=0, enable_tuner=0, compile_options={'test': 233}):
        cy = test_func(x, w, stride, padding, dilation, groups)
        (cdx, cdw) = jt.grad(cy, [x, w])
        jt.sync([cy, cdx, cdw])
    logs = find_log_with_re(raw_log, '(Jit op key (not )?found: .*conv.*)')
    assert len(logs) == 3
    assert np.allclose(y.data, cy.data)
    assert np.allclose(dw.data, cdw.data, 0.001), (dw.data, cdw.data, np.abs(dw.data - cdw.data).max())
    assert np.allclose(dx.data, cdx.data, 0.001), (dx.data, cdx.data, np.abs(dx.data - cdx.data).max())

class TestGroupConvTuner(unittest.TestCase):

    @unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
    def test_forward_cuda(self):
        if False:
            print('Hello World!')
        for groups in [2, 4, 8]:
            check_forward([10, 8, 100, 100], [8, 8 // groups, 3, 3], 1, 0, 1, groups, 1, False)
            check_forward([10, 8, 40, 50], [16, 8 // groups, 5, 5], 1, 1, 2, groups, 1, False)
            check_forward([10, 8, 40, 50], [16, 8 // groups, 4, 4], 3, 1, 3, groups, 1, False)

    def test_forward(self):
        if False:
            while True:
                i = 10
        for groups in [2, 4, 8]:
            check_forward([10, 8, 100, 100], [8, 8 // groups, 3, 3], 1, 0, 1, groups, 0, False)
            check_forward([10, 8, 40, 50], [16, 8 // groups, 5, 5], 1, 1, 2, groups, 0, False)
            check_forward([10, 8, 40, 50], [16, 8 // groups, 4, 4], 3, 1, 3, groups, 0, False)

    @unittest.skipIf(not jt.compiler.has_cuda, 'No CUDA found')
    def test_backward_cuda(self):
        if False:
            i = 10
            return i + 15
        for groups in [2, 4, 8]:
            check_backward([10, 8, 100, 100], [8, 8 // groups, 3, 3], 1, 0, 1, groups, 1, False)
            check_backward([10, 8, 40, 50], [16, 8 // groups, 5, 5], 1, 1, 2, groups, 1, False)
            check_backward([10, 8, 40, 50], [16, 8 // groups, 4, 4], 3, 1, 3, groups, 1, False)

    def test_backward(self):
        if False:
            while True:
                i = 10
        for groups in [2, 4, 8]:
            check_backward([10, 8, 100, 100], [8, 8 // groups, 3, 3], 1, 0, 1, groups, 0, False)
            check_backward([10, 8, 40, 50], [16, 8 // groups, 5, 5], 1, 1, 2, groups, 0, False)
            check_backward([10, 8, 40, 50], [16, 8 // groups, 4, 4], 3, 1, 3, groups, 0, False)
if __name__ == '__main__':
    unittest.main()