import numpy
from chainer import function_node
import chainer.functions
from chainer.utils import type_check

class MeanSquaredError(function_node.FunctionNode):
    """Mean squared error (a.k.a. Euclidean loss) function."""

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(in_types[0].dtype.kind == 'f', in_types[0].dtype == in_types[1].dtype, in_types[0].shape == in_types[1].shape)

    def forward_cpu(self, inputs):
        if False:
            i = 10
            return i + 15
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        return (numpy.array(diff.dot(diff) / diff.size, dtype=diff.dtype),)

    def forward_gpu(self, inputs):
        if False:
            while True:
                i = 10
        self.retain_inputs((0, 1))
        diff = (inputs[0] - inputs[1]).ravel()
        return (diff.dot(diff) / diff.dtype.type(diff.size),)

    def backward(self, indexes, gy):
        if False:
            return 10
        (x0, x1) = self.get_retained_inputs()
        ret = []
        diff = x0 - x1
        gy0 = chainer.functions.broadcast_to(gy[0], diff.shape)
        gx0 = gy0 * diff * (2.0 / diff.size)
        if 0 in indexes:
            ret.append(gx0)
        if 1 in indexes:
            ret.append(-gx0)
        return ret

def mean_squared_error(x0, x1):
    if False:
        for i in range(10):
            print('nop')
    'Mean squared error function.\n\n    The function computes the mean squared error between two variables. The\n    mean is taken over the minibatch. Args ``x0`` and ``x1`` must have the\n    same dimensions. Note that the error is not scaled by 1/2.\n\n    Args:\n        x0 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n        x1 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable:\n            A variable holding an array representing the mean squared\n            error of two inputs.\n\n     .. admonition:: Example\n\n        1D array examples:\n\n        >>> x = np.array([1, 2, 3, 4]).astype(np.float32)\n        >>> y = np.array([0, 0, 0, 0]).astype(np.float32)\n        >>> F.mean_squared_error(x, y)\n        variable(7.5)\n        >>> x = np.array([1, 2, 3, 4, 5, 6]).astype(np.float32)\n        >>> y = np.array([7, 8, 9, 10, 11, 12]).astype(np.float32)\n        >>> F.mean_squared_error(x, y)\n        variable(36.)\n\n        2D array example:\n\n        In this example, there are 4 elements, and thus 4 errors\n        >>> x = np.array([[1, 2], [3, 4]]).astype(np.float32)\n        >>> y = np.array([[8, 8], [8, 8]]).astype(np.float32)\n        >>> F.mean_squared_error(x, y)\n        variable(31.5)\n\n        3D array example:\n\n        In this example, there are 8 elements, and thus 8 errors\n        >>> x = np.reshape(np.array([1, 2, 3, 4, 5, 6, 7, 8]), (2, 2, 2))\n        >>> y = np.reshape(np.array([8, 8, 8, 8, 8, 8, 8, 8]), (2, 2, 2))\n        >>> x = x.astype(np.float32)\n        >>> y = y.astype(np.float32)\n        >>> F.mean_squared_error(x, y)\n        variable(17.5)\n\n    '
    return MeanSquaredError().apply((x0, x1))[0]