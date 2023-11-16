from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class FlipUD(function_node.FunctionNode):
    """Flip array in the up/down direction."""

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('a',))
        a_type = in_types[0]
        type_check.expect(a_type.dtype.kind == 'f', a_type.ndim >= 1)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        xp = backend.get_array_module(*inputs)
        return (xp.flipud(inputs[0]),)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        return FlipUD().apply(grad_outputs)

def flipud(a):
    if False:
        while True:
            i = 10
    'Flip array in the up/down direction.\n\n    Args:\n        a (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return FlipUD().apply((a,))[0]