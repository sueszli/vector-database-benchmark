import numpy
import six
import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class Dstack(function_node.FunctionNode):
    """Concatenate multiple tensors along third axis (depth wise)."""

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check.expect(in_types.size() > 0)
        type_check._argname((in_types[0],), ('x0',))
        ndim = type_check.eval(in_types[0].ndim)
        for i in six.moves.range(1, type_check.eval(in_types.size())):
            type_check._argname((in_types[i],), ('x{}'.format(i),))
            type_check.expect(in_types[0].dtype == in_types[i].dtype, in_types[0].ndim == in_types[i].ndim)
            if ndim <= 2:
                type_check.expect(in_types[0].shape == in_types[i].shape)
                continue
            for d in six.moves.range(0, ndim):
                if d == 2:
                    continue
                type_check.expect(in_types[0].shape[d] == in_types[i].shape[d])

    def forward(self, xs):
        if False:
            for i in range(10):
                print('nop')
        xp = backend.get_array_module(*xs)
        return (xp.dstack(xs),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (gy,) = grad_outputs
        ndim = len(self.inputs[0].shape)
        if len(self.inputs) == 1:
            if ndim <= 2:
                return (gy.reshape(self.inputs[0].shape),)
            return (gy,)
        if ndim <= 2:
            gxs = chainer.functions.split_axis(gy, len(self.inputs), axis=2)
            return [gx.reshape(self.inputs[0].shape) for gx in gxs]
        sizes = numpy.array([x.shape[2] for x in self.inputs[:-1]]).cumsum()
        return chainer.functions.split_axis(gy, sizes, axis=2)

def dstack(xs):
    if False:
        return 10
    "Concatenate variables along third axis (depth wise).\n\n    Args:\n        xs (list of :class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variables to be concatenated. The variables must have the\n            same ``ndim``. When the variables have the third axis (i.e.\n            :math:`ndim \\geq 3`), the variables must have the same shape\n            along all but the third axis. When the variables do not have the\n            third axis(i.e. :math:`ndim < 3`), the variables must have the\n            same shape.\n\n    Returns:\n        ~chainer.Variable:\n            Output variable. When the input variables have the third axis\n            (i.e. :math:`ndim \\geq 3`), the shapes of inputs and output are\n            the same along all but the third axis. The length of third axis\n            is the sum of the lengths of inputs' third axis.\n            When the shape of variables are ``(N1, N2)`` (i.e.\n            :math:`ndim = 2`), the shape of output is ``(N1, N2, 2)``. When\n            the shape of variables are ``(N1,)`` (i.e. :math:`ndim = 1`), the\n            shape of output is ``(1, N1, 2)``. When the shape of variables are\n            ``()`` (i.e. :math:`ndim = 0`), the shape of output is\n            ``(1, 1, 2)``.\n\n\n    .. admonition:: Example\n\n        >>> x1 = np.array((1, 2, 3))\n        >>> x1.shape\n        (3,)\n        >>> x2 = np.array((2, 3, 4))\n        >>> x2.shape\n        (3,)\n        >>> y = F.dstack((x1, x2))\n        >>> y.shape\n        (1, 3, 2)\n        >>> y.array\n        array([[[1, 2],\n                [2, 3],\n                [3, 4]]])\n\n        >>> x1 = np.arange(0, 6).reshape(3, 2)\n        >>> x1.shape\n        (3, 2)\n        >>> x1\n        array([[0, 1],\n               [2, 3],\n               [4, 5]])\n        >>> x2 = np.arange(6, 12).reshape(3, 2)\n        >>> x2.shape\n        (3, 2)\n        >>> x2\n        array([[ 6,  7],\n               [ 8,  9],\n               [10, 11]])\n        >>> y = F.dstack([x1, x2])\n        >>> y.shape\n        (3, 2, 2)\n        >>> y.array\n        array([[[ 0,  6],\n                [ 1,  7]],\n        <BLANKLINE>\n               [[ 2,  8],\n                [ 3,  9]],\n        <BLANKLINE>\n               [[ 4, 10],\n                [ 5, 11]]])\n\n        >>> x1 = np.arange(0, 12).reshape(3, 2, 2)\n        >>> x2 = np.arange(12, 18).reshape(3, 2, 1)\n        >>> y = F.dstack([x1, x2])\n        >>> y.shape\n        (3, 2, 3)\n        >>> y.array\n        array([[[ 0,  1, 12],\n                [ 2,  3, 13]],\n        <BLANKLINE>\n               [[ 4,  5, 14],\n                [ 6,  7, 15]],\n        <BLANKLINE>\n               [[ 8,  9, 16],\n                [10, 11, 17]]])\n\n    "
    return Dstack().apply(xs)[0]