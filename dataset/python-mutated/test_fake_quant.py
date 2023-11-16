import numpy as np
import pytest
import megengine as mge
import megengine.functional as F
from megengine import tensor
from megengine.core.autodiff.grad import Function, Grad
from megengine.core.tensor.dtype import QuantDtypeMeta
from megengine.core.tensor.utils import make_shape_tuple
from megengine.quantization.internal_fake_quant import *
from megengine.quantization.utils import QuantMode, create_qparams, fake_quant_tensor, lsq_forward, tqt_forward

class TQT_numpy:

    def __init__(self, lowerbound, upperbound):
        if False:
            for i in range(10):
                print('nop')
        super().__init__()
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def forward(self, inp, scale):
        if False:
            return 10
        t = 2 ** scale
        inp_scaled = inp / t
        inp_clipped = np.maximum(np.minimum(inp_scaled, self.upperbound), self.lowerbound)
        inp_rounded = np.round(inp_clipped)
        inp_flq = inp_rounded * t
        self.saved_tensors = (inp_scaled, inp_rounded, t)
        return inp_flq

    def backward(self, grad_inp_flq):
        if False:
            print('Hello World!')
        (inp_scaled, inp_rounded, t) = self.saved_tensors
        mask_clip = (inp_scaled < -0.5 + self.lowerbound) + (inp_scaled > self.upperbound + 0.5)
        mask_quant = np.abs(mask_clip - 1)
        grad_quant = grad_inp_flq * mask_quant * (inp_rounded - inp_scaled)
        grad_clip = grad_inp_flq * mask_clip * inp_rounded
        grad_s = grad_clip.sum() + grad_quant.sum()
        grad_s = grad_s * t * np.log(2)
        grad_inp = grad_inp_flq * mask_quant
        return (grad_inp, grad_s)

def test_tqt():
    if False:
        while True:
            i = 10
    g = []

    def cb(grad):
        if False:
            print('Hello World!')
        g.append(grad)
    x = np.random.randint(-128, 128, size=(1, 2, 3, 4)).astype('float32')
    s = np.random.rand(1) - 1
    g_y = np.ones(shape=(1, 2, 3, 4), dtype='float32')
    n = TQT_numpy(-127, 127)
    y_np = n.forward(x, s)
    (g_x_np, g_s_np) = n.backward(g_y)
    x = mge.tensor(x, dtype='float32')
    s = mge.tensor(s, dtype='float32')
    g_y = mge.tensor(g_y, dtype='float32')
    with Grad() as grad:
        grad.wrt(x, s, callback=cb)
        y = tqt_forward(-127, 127, x, s)
        grad(y, g_y)
    (g_x, g_s) = g
    np.testing.assert_allclose(y.numpy(), y_np, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(g_x.numpy(), g_x_np, rtol=1e-05, atol=1e-05)
    np.testing.assert_allclose(g_s.numpy(), g_s_np, rtol=5e-05, atol=5e-05)

def _save_to(self, name='grad'):
    if False:
        print('Hello World!')

    def callback(grad):
        if False:
            while True:
                i = 10
        setattr(self, name, grad)
    return callback

class Round(Function):

    def forward(self, x):
        if False:
            print('Hello World!')
        return F.round(x)

    def backward(self, output_grads):
        if False:
            return 10
        return output_grads

def fake_quant_tensor_gt(inp, scale, zero_point, qmin, qmax):
    if False:
        print('Hello World!')
    oup = Round()(inp / scale) + zero_point
    oup = F.minimum(F.maximum(oup, qmin), qmax)
    oup = (oup - zero_point) * scale
    return oup

def test_fakequant():
    if False:
        print('Hello World!')
    qmin = -126
    qmax = 129
    test_dtype = QuantDtypeMeta('test_qint8', None, 'int8', qmin, qmax)

    def run(zero_point, scale):
        if False:
            i = 10
            return i + 15
        qparams = create_qparams(QuantMode.ASYMMERTIC, test_dtype, scale, zero_point)
        inp_data = np.random.uniform(low=-512.0, high=512.0, size=(1, 32, 32, 32))
        inp = tensor(inp_data, dtype=np.float32)
        oup = fake_quant_tensor(inp, qparams).numpy()
        oup_gt = fake_quant_tensor_gt(inp, scale, zero_point, qmin, qmax).numpy()
        assert np.allclose(oup, oup_gt)
        assert oup.shape == oup_gt.shape
        x = tensor(inp_data, dtype=np.float32)
        with Grad() as grad:
            grad.wrt(x, callback=_save_to(x))
            y = fake_quant_tensor(x, qparams)
            grad(y, tensor(F.ones_like(x)))
        x1 = tensor(inp_data, dtype=np.float32)
        with Grad() as grad:
            grad.wrt(x1, callback=_save_to(x1))
            y1 = fake_quant_tensor_gt(x1, scale, zero_point, qmin, qmax)
            grad(y1, tensor(F.ones_like(x1)))
        assert np.allclose(x.grad.numpy(), x1.grad.numpy())
        assert make_shape_tuple(x.grad.shape) == make_shape_tuple(x1.grad.shape)
        x = F.full((1, 32, 3, 3), np.nan)
        y = fake_quant_tensor(x, qparams).numpy()
        assert np.isnan(y).all()
    zero_point = tensor([1.0], dtype=np.float32)
    scale = tensor([4.0], dtype=np.float32)
    run(zero_point, scale)
    zero_point = tensor(1.0 * np.ones((1, 32, 1, 1)), dtype=np.float32)
    scale = tensor(4.0 * np.ones((1, 32, 1, 1)), dtype=np.float32)
    run(zero_point, scale)

class LSQ_numpy:

    def __init__(self, lowerbound, upperbound):
        if False:
            return 10
        super().__init__()
        self.lowerbound = lowerbound
        self.upperbound = upperbound

    def forward(self, inp, scale, zero_point, grad_scale):
        if False:
            return 10
        inp_scaled = inp / scale + zero_point
        inp_clipped = np.maximum(np.minimum(inp_scaled, self.upperbound), self.lowerbound)
        inp_rounded = np.floor(inp_clipped + 0.5)
        inp_flq = (inp_rounded - zero_point) * scale
        self.saved_tensors = (inp_scaled, inp_rounded, scale, grad_scale)
        return inp_flq

    def backward(self, grad_inp_flq):
        if False:
            return 10
        (inp_scaled, inp_rounded, scale, grad_scale) = self.saved_tensors
        ind_small = inp_scaled < self.lowerbound
        ind_big = inp_scaled > self.upperbound
        ind_middle = np.logical_xor(ind_small, ind_big)
        ind_middle = np.abs(ind_middle - 1)
        grad_s = ind_small * self.lowerbound + ind_big * self.upperbound + ind_middle * (-inp_scaled + inp_rounded)
        grad_s = grad_s * grad_scale * grad_inp_flq
        grad_s = grad_s.sum()
        grad_inp = grad_inp_flq * ind_middle
        return (grad_inp, grad_s)

def test_lsq():
    if False:
        while True:
            i = 10
    g = []

    def cb(grad):
        if False:
            i = 10
            return i + 15
        g.append(grad)
    x = np.array([[[[4.0, 38.0, -121.0, 38.0], [15.0, -115.0, -112.0, 24.0], [23.0, -65.0, 109.0, -115.0]], [[-66.0, -90.0, -45.0, -101.0], [68.0, -98.0, 108.0, -79.0], [54.0, 63.0, -10.0, -50.0]]]], dtype='float32')
    s = np.array([0.02918224], dtype='float32')
    eps = np.array([1e-05], dtype='float32')
    s = np.abs(s) if np.abs(s) > eps else eps
    zero_point = np.array([1.0], dtype='float32')
    grad_s = np.array([2.0], dtype='float32')
    g_y = np.ones(shape=(1, 2, 3, 4), dtype='float32')
    n = LSQ_numpy(-127, 127)
    y_np = n.forward(x, s, zero_point, grad_s)
    (g_x_np, g_s_np) = n.backward(g_y)
    x = mge.tensor(x, dtype='float32')
    s = mge.tensor(s, dtype='float32')
    zero_point = mge.tensor(zero_point, dtype='float32')
    grad_s = mge.tensor(grad_s, dtype='float32')
    g_y = mge.tensor(g_y, dtype='float32')
    with Grad() as grad:
        grad.wrt(x, s, callback=cb)
        y = lsq_forward(-127, 127, x, s, zero_point, grad_s)
        grad(y, g_y)
    (g_x, g_s) = g
    np.testing.assert_allclose(y.numpy(), y_np, rtol=1e-07, atol=1e-07)
    np.testing.assert_allclose(g_x.numpy(), g_x_np, rtol=1e-07, atol=1e-07)
    np.testing.assert_allclose(g_s.numpy(), g_s_np, rtol=5e-07, atol=5e-07)