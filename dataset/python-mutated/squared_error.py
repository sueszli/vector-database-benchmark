from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class SquaredError(function_node.FunctionNode):
    """Squared error function."""

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(in_types[0].dtype.kind == 'f', in_types[0].dtype == in_types[1].dtype, in_types[0].shape == in_types[1].shape)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        (x0, x1) = inputs
        diff = x0 - x1
        self.retain_inputs((0, 1))
        return (utils.force_array(diff * diff, dtype=x0.dtype),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (x0, x1) = self.get_retained_inputs()
        (gy,) = grad_outputs
        gx = gy * 2 * (x0 - x1)
        return (gx, -gx)

def squared_error(x0, x1):
    if False:
        i = 10
        return i + 15
    'Squared error function.\n\n    This function computes the squared error between two variables:\n\n    .. math::\n\n        (x_0 - x_1)^2\n\n    where operation is done in elementwise manner.\n    Note that the error is not scaled by 1/2:\n\n    Args:\n        x0 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n        x1 (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable:\n            A variable holding an array representing the squared error of\n            two inputs.\n\n    .. note::\n\n        :func:`~chainer.functions.squared_error` and\n        :func:`~chainer.functions.squared_difference` are identical functions,\n        aside from the different argument names.\n        They are both kept for backward compatibility.\n\n    .. seealso:: :func:`~chainer.functions.squared_difference`\n\n    .. admonition:: Example\n\n        >>> x1 = np.arange(6).astype(np.float32)\n        >>> x1\n        array([0., 1., 2., 3., 4., 5.], dtype=float32)\n        >>> x2 = np.array([5, 4, 3, 2, 1, 0]).astype(np.float32)\n        >>> x2\n        array([5., 4., 3., 2., 1., 0.], dtype=float32)\n        >>> y = F.squared_error(x1, x2)\n        >>> y.shape\n        (6,)\n        >>> y.array\n        array([25.,  9.,  1.,  1.,  9., 25.], dtype=float32)\n\n    .. seealso:: :func:`~chainer.functions.squared_difference`\n\n    '
    return SquaredError().apply((x0, x1))[0]

def squared_difference(x1, x2):
    if False:
        while True:
            i = 10
    'Squared difference function.\n\n    This functions is identical to :func:`~chainer.functions.squared_error`\n    except for the names of the arguments.\n\n    .. seealso:: :func:`~chainer.functions.squared_error`\n\n    '
    return squared_error(x1, x2)