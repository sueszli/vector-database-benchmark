import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class Space2Depth(function_node.FunctionNode):
    """Space to depth transformation."""

    def __init__(self, r):
        if False:
            return 10
        self.r = r

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f', in_types[0].ndim == 4)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        (X,) = inputs
        xp = backend.get_array_module(X)
        (bsize, c, a, b) = X.shape
        X = xp.reshape(X, (bsize, c, a // self.r, self.r, b // self.r, self.r))
        X = xp.transpose(X, (0, 3, 5, 1, 2, 4))
        X = xp.reshape(X, (bsize, self.r ** 2 * c, a // self.r, b // self.r))
        return (X,)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (gy,) = grad_outputs
        gy = chainer.functions.depth2space(gy, self.r)
        return (gy,)

def space2depth(X, r):
    if False:
        while True:
            i = 10
    'Computes the space2depth transformation for subpixel calculations.\n\n    Args:\n        X (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Variable holding a 4d array of shape\n            ``(batch, channel, dim1 * r, dim2 * r)``.\n        r (int): the downscaling factor.\n\n    Returns:\n        ~chainer.Variable:\n            A variable holding the downscaled layer array from subpixel array\n            sampling. The shape is ``(batch, channel * r * r, dim1, dim2)``.\n\n    .. note::\n       This can be used to compute inverse super-resolution transformations.\n       See https://arxiv.org/abs/1609.05158 for details.\n\n    .. seealso:: :func:`depth2space`\n\n    .. admonition:: Example\n\n        >>> X = np.arange(24).reshape(1, 1, 4, 6).astype(np.float32)\n        >>> X.shape\n        (1, 1, 4, 6)\n        >>> X\n        array([[[[ 0.,  1.,  2.,  3.,  4.,  5.],\n                 [ 6.,  7.,  8.,  9., 10., 11.],\n                 [12., 13., 14., 15., 16., 17.],\n                 [18., 19., 20., 21., 22., 23.]]]], dtype=float32)\n        >>> y = F.space2depth(X, 2)\n        >>> y.shape\n        (1, 4, 2, 3)\n        >>> y.array\n        array([[[[ 0.,  2.,  4.],\n                 [12., 14., 16.]],\n        <BLANKLINE>\n                [[ 1.,  3.,  5.],\n                 [13., 15., 17.]],\n        <BLANKLINE>\n                [[ 6.,  8., 10.],\n                 [18., 20., 22.]],\n        <BLANKLINE>\n                [[ 7.,  9., 11.],\n                 [19., 21., 23.]]]], dtype=float32)\n\n    '
    return Space2Depth(r).apply((X,))[0]