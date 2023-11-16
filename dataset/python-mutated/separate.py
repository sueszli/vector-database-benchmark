from chainer import backend
from chainer import function_node
from chainer.functions.array import stack
from chainer.utils import type_check

class Separate(function_node.FunctionNode):
    """Function that separates a given array."""

    def __init__(self, axis):
        if False:
            i = 10
            return i + 15
        self.axis = axis

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        if self.axis >= 0:
            type_check.expect(self.axis < x_type.ndim)
        else:
            type_check.expect(-self.axis <= x_type.ndim)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        self._xp = backend.get_array_module(x)
        xs = self._xp.split(x, x.shape[self.axis], self.axis)
        ys = [self._xp.squeeze(y, self.axis) for y in xs]
        self._shape = ys[0].shape
        self._dtype = x.dtype
        return tuple(ys)

    def backward(self, indexes, grad_outputs):
        if False:
            i = 10
            return i + 15
        grad_outputs = [self._xp.zeros(self._shape, dtype=self._dtype) if g is None else g for g in grad_outputs]
        return (stack.stack(grad_outputs, self.axis),)

def separate(x, axis=0):
    if False:
        while True:
            i = 10
    'Separates an array along a given axis.\n\n    This function separates an array along a given axis. For example, shape of\n    an array is ``(2, 3, 4)``. When it separates the array with ``axis=1``, it\n    returns three ``(2, 4)`` arrays.\n\n    This function is an inverse of :func:`chainer.functions.stack`.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Variable to be separated.\n            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.\n        axis (int): Axis along which variables are separated.\n\n    Returns:\n        tuple of chainer.Variable: Output variables.\n\n    .. seealso:: :func:`chainer.functions.stack`\n\n    .. admonition:: Example\n\n        >>> x = np.arange(6).reshape((2, 3)).astype(np.float32)\n        >>> x\n        array([[0., 1., 2.],\n               [3., 4., 5.]], dtype=float32)\n        >>> x.shape\n        (2, 3)\n        >>> y = F.separate(x) # split along axis=0\n        >>> isinstance(y, tuple)\n        True\n        >>> len(y)\n        2\n        >>> y[0].shape\n        (3,)\n        >>> y[0].array\n        array([0., 1., 2.], dtype=float32)\n        >>> y = F.separate(x, axis=1)\n        >>> len(y)\n        3\n        >>> y[0].shape\n        (2,)\n        >>> y[0].array\n        array([0., 3.], dtype=float32)\n\n    '
    return Separate(axis).apply((x,))