import numpy as np
import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.utils import argument
from chainer.utils import type_check

def _kern():
    if False:
        for i in range(10):
            print('nop')
    return cuda.elementwise('T cond, T x, T slope', 'T y', 'y = cond >= 0 ? x : (T)(slope * x)', 'rrelu')

class RReLU(function_node.FunctionNode):
    """Randomized Leaky rectifier unit."""

    def __init__(self, lower=1.0 / 8, upper=1.0 / 3, r=None):
        if False:
            return 10
        if not 0.0 <= lower < 1.0:
            raise ValueError('lower must be in the range [0, 1)')
        if not 0.0 <= upper < 1.0:
            raise ValueError('upper must be in the range [0, 1)')
        if not lower < upper:
            raise ValueError('lower must be less than upper')
        self.lower = lower
        self.upper = upper
        self.r = r

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check.expect(in_types.size() == 1)
        (x_type,) = in_types
        type_check.expect(x_type.dtype.kind == 'f')
        if self.r is not None:
            type_check.expect(x_type.dtype == self.r.dtype)
            type_check.expect(x_type.shape == self.r.shape)

    def forward_cpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        if chainer.config.train:
            if self.r is None:
                self.r = np.random.uniform(self.lower, self.upper, x.shape).astype(x.dtype, copy=False)
        else:
            self.r = np.full(x.shape, (self.lower + self.upper) / 2, dtype=x.dtype)
        y = np.where(x >= 0, x, x * self.r)
        self.retain_outputs((0,))
        return (y,)

    def forward_gpu(self, inputs):
        if False:
            return 10
        (x,) = inputs
        xp = cuda.cupy
        if chainer.config.train:
            if self.r is None:
                self.r = xp.random.uniform(self.lower, self.upper, x.shape).astype(x.dtype, copy=False)
        else:
            self.r = xp.full(x.shape, (self.lower + self.upper) / 2, dtype=x.dtype)
        y = _kern()(x, x, self.r)
        self.retain_outputs((0,))
        return (y,)

    def backward(self, indexes, grad_outputs):
        if False:
            i = 10
            return i + 15
        y = self.get_retained_outputs()[0].data
        return _RReLUGrad(y, self.r).apply(grad_outputs)

class _RReLUGrad(function_node.FunctionNode):

    def __init__(self, y, r):
        if False:
            return 10
        self.r = r
        self.y = y

    def forward_cpu(self, inputs):
        if False:
            i = 10
            return i + 15
        (gy,) = inputs
        gy = np.where(self.y >= 0, gy, gy * self.r)
        return (gy,)

    def forward_gpu(self, inputs):
        if False:
            return 10
        (gy,) = inputs
        gy = _kern()(self.y, gy, self.r)
        return (gy,)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        return _RReLUGrad(self.y, self.r).apply(grad_outputs)

def rrelu(x, l=1.0 / 8, u=1.0 / 3, **kwargs):
    if False:
        while True:
            i = 10
    'rrelu(x, l=1. / 8, u=1. / 3, *, r=None, return_r=False)\n\n    Randomized Leaky Rectified Liner Unit function.\n\n    This function is expressed as\n\n    .. math:: f(x)=\\max(x, rx),\n\n    where :math:`r` is a random number sampled from a uniform distribution\n    :math:`U(l, u)`.\n\n    .. note::\n\n        The :math:`r` corresponds to :math:`a` in the original\n        paper (https://arxiv.org/pdf/1505.00853.pdf).\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.\n        l (float): The lower bound of the uniform distribution.\n        u (float): The upper bound of the uniform distribution.\n        r (:ref:`ndarray` or None):\n            The r to be used for rrelu.\n            The shape and dtype must be the same as ``x[0]`` and should be on\n            the same device.\n            If ``r``  is not specified or set to ``None``, an ``r`` will be\n            generated randomly according to the given ``l`` and ``u``.\n            If ``r`` is specified, ``l`` and ``u`` will be ignored.\n        return_r (bool):\n            If ``True``, the r used for rrelu is returned altogether with\n            the output variable.\n            The returned ``r`` can latter be reused by passing it to ``r``\n            argument.\n\n    Returns:\n        ~chainer.Variable or tuple:\n            When ``return_r`` is ``False`` (default), return the output\n            variable. Otherwise returnes the tuple of the output variable and\n            ``r`` (:ref:`ndarray`). The ``r`` will be on the same device as\n            the input.\n            A :math:`(s_1, s_2, ..., s_N)`-shaped float array.\n\n    .. admonition:: Example\n\n        >>> x = np.array([[-1, 0], [2, -3], [-2, 1]], np.float32)\n        >>> x\n        array([[-1.,  0.],\n               [ 2., -3.],\n               [-2.,  1.]], dtype=float32)\n        >>> F.rrelu(x).array # doctest: +SKIP\n        array([[-0.24850948,  0.        ],\n               [ 2.        , -0.50844127],\n               [-0.598535  ,  1.        ]], dtype=float32)\n    '
    r = None
    return_r = False
    if kwargs:
        (r, return_r) = argument.parse_kwargs(kwargs, ('r', r), ('return_r', r), train='train argument is not supported anymore. Use chainer.using_config')
    func = RReLU(l, u, r)
    (out,) = func.apply((x,))
    r = func.r
    if return_r:
        return (out, r)
    return out