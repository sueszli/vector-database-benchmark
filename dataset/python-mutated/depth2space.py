import numpy
import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class Depth2Space(function_node.FunctionNode):
    """Depth to space transformation."""

    def __init__(self, r):
        if False:
            while True:
                i = 10
        self.r = r

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f', in_types[0].ndim == 4)

    def forward(self, inputs):
        if False:
            return 10
        (X,) = inputs
        xp = backend.get_array_module(X)
        (bsize, c, a, b) = X.shape
        c //= self.r ** 2
        if xp is numpy:
            X = xp.transpose(X, (0, 2, 3, 1))
            X = xp.reshape(X, (bsize, a, b, self.r, self.r, c))
            X = xp.transpose(X, (0, 1, 3, 2, 4, 5))
            X = xp.reshape(X, (bsize, a * self.r, b * self.r, c))
            X = xp.transpose(X, (0, 3, 1, 2))
        else:
            X = xp.reshape(X, (bsize, self.r, self.r, c, a, b))
            X = xp.transpose(X, (0, 3, 4, 1, 5, 2))
            X = xp.reshape(X, (bsize, c, a * self.r, b * self.r))
        return (X,)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (gy,) = grad_outputs
        gy = chainer.functions.space2depth(gy, self.r)
        return (gy,)

def depth2space(X, r):
    if False:
        for i in range(10):
            print('nop')
    'Computes the depth2space transformation for subpixel calculations.\n\n    Args:\n        X (:class:`~chainer.Variable` or :ref:`ndarray`): Variable holding a\n            4d array of shape ``(batch, channel * r * r, dim1, dim2)``.\n        r (int): the upscaling factor.\n\n    Returns:\n        ~chainer.Variable:\n            A variable holding the upscaled array from\n            interspersed depth layers. The shape is\n            ``(batch, channel, dim1 * r, dim2 * r)``.\n\n    .. note::\n       This can be used to compute super-resolution transformations.\n       See https://arxiv.org/abs/1609.05158 for details.\n\n    .. seealso:: :func:`space2depth`\n\n    .. admonition:: Example\n\n        >>> X = np.arange(24).reshape(1, 4, 2, 3).astype(np.float32)\n        >>> X.shape\n        (1, 4, 2, 3)\n        >>> X\n        array([[[[ 0.,  1.,  2.],\n                 [ 3.,  4.,  5.]],\n        <BLANKLINE>\n                [[ 6.,  7.,  8.],\n                 [ 9., 10., 11.]],\n        <BLANKLINE>\n                [[12., 13., 14.],\n                 [15., 16., 17.]],\n        <BLANKLINE>\n                [[18., 19., 20.],\n                 [21., 22., 23.]]]], dtype=float32)\n        >>> y = F.depth2space(X, 2)\n        >>> y.shape\n        (1, 1, 4, 6)\n        >>> y.array\n        array([[[[ 0.,  6.,  1.,  7.,  2.,  8.],\n                 [12., 18., 13., 19., 14., 20.],\n                 [ 3.,  9.,  4., 10.,  5., 11.],\n                 [15., 21., 16., 22., 17., 23.]]]], dtype=float32)\n\n    '
    return Depth2Space(r).apply((X,))[0]