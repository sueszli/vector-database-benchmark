import six
import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class CReLU(function_node.FunctionNode):
    """Concatenated Rectified Linear Unit."""

    def __init__(self, axis=1):
        if False:
            i = 10
            return i + 15
        if not isinstance(axis, six.integer_types):
            raise TypeError('axis must be an integer value')
        self.axis = axis

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f', in_types[0].ndim > self.axis, in_types[0].ndim >= -self.axis)

    def get_output_shape(self, input_shape):
        if False:
            for i in range(10):
                print('nop')
        output_shape = list(input_shape)
        output_shape[self.axis] *= 2
        return tuple(output_shape)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        xp = backend.get_array_module(x)
        y = xp.empty(self.get_output_shape(x.shape), dtype=x.dtype)
        (y_former, y_latter) = xp.split(y, 2, axis=self.axis)
        zero = x.dtype.type(0)
        xp.maximum(zero, x, out=y_former)
        xp.maximum(zero, -x, out=y_latter)
        self.retain_inputs((0,))
        return (y,)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (x,) = self.get_retained_inputs()
        (gy,) = grad_outputs
        (gy_former, gy_latter) = chainer.functions.split_axis(gy, 2, axis=self.axis)
        return (gy_former * (x.data > 0) - gy_latter * (x.data < 0),)

def crelu(x, axis=1):
    if False:
        i = 10
        return i + 15
    'Concatenated Rectified Linear Unit function.\n\n    This function is expressed as follows\n\n     .. math:: f(x) = (\\max(0, x), \\max(0, -x)).\n\n    Here, two output values are concatenated along an axis.\n\n    See: https://arxiv.org/abs/1603.05201\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.\n        axis (int): Axis that the output values are concatenated along.\n            Default is 1.\n\n    Returns:\n        ~chainer.Variable: Output variable of concatenated array.\n        If the axis is 1, A :math:`(s_1, s_2 \\times 2, ..., s_N)`-shaped float\n        array.\n\n    .. admonition:: Example\n\n        >>> x = np.array([[-1, 0], [2, -3]], np.float32)\n        >>> x\n        array([[-1.,  0.],\n               [ 2., -3.]], dtype=float32)\n        >>> y = F.crelu(x, axis=1)\n        >>> y.array\n        array([[0., 0., 1., 0.],\n               [2., 0., 0., 3.]], dtype=float32)\n\n    '
    return CReLU(axis=axis).apply((x,))[0]