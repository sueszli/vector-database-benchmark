from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx

class AbsoluteError(function_node.FunctionNode):
    """Element-wise absolute error function."""

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('x0', 'x1'))
        type_check.expect(in_types[0].dtype.kind == 'f', in_types[0].dtype == in_types[1].dtype, in_types[0].shape == in_types[1].shape)

    def forward_chainerx(self, inputs):
        if False:
            print('Hello World!')
        (x0, x1) = inputs
        self.diff = x0 - x1
        return (chainerx.abs(self.diff),)

    def forward(self, inputs):
        if False:
            return 10
        (x0, x1) = inputs
        self.diff = x0 - x1
        return (utils.force_array(abs(self.diff), dtype=x0.dtype),)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        (gy,) = grad_outputs
        gx = gy * backend.get_array_module(gy).sign(self.diff)
        return (gx, -gx)

def absolute_error(x0, x1):
    if False:
        i = 10
        return i + 15
    'Element-wise absolute error function.\n\n    Computes the element-wise absolute error :math:`L` between two inputs\n    :math:`x_0` and :math:`x_1` defined as follows.\n\n    .. math::\n\n        L = |x_0 - x_1|\n\n    Args:\n        x0 (:class:`~chainer.Variable` or :ref:`ndarray`):\n            First input variable.\n        x1 (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Second input variable.\n\n    Returns:\n        ~chainer.Variable:\n            An array representing the element-wise absolute error between the\n            two inputs.\n\n    '
    return AbsoluteError().apply((x0, x1))[0]