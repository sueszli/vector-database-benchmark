from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class Arctanh(function_node.FunctionNode):
    """Elementwise inverse hyperbolic tangent function."""

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('x',))
        (x_type,) = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, inputs):
        if False:
            print('Hello World!')
        self.retain_inputs((0,))
        (x,) = inputs
        xp = backend.get_array_module(x)
        y = xp.arctanh(x)
        return (utils.force_array(y, dtype=x.dtype),)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        (x,) = self.get_retained_inputs()
        (gy,) = grad_outputs
        gx = 1.0 / (1 - x ** 2) * gy
        return (gx,)

def arctanh(x):
    if False:
        print('Hello World!')
    'Elementwise inverse hyperbolic tangent function.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return Arctanh().apply((x,))[0]