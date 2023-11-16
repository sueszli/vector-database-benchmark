from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class FlipLR(function_node.FunctionNode):
    """Flip array in the left/right direction."""

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check._argname(in_types, ('a',))
        a_type = in_types[0]
        type_check.expect(a_type.dtype.kind == 'f', a_type.ndim >= 2)

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        xp = backend.get_array_module(*inputs)
        return (xp.fliplr(inputs[0]),)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        return FlipLR().apply(grad_outputs)

def fliplr(a):
    if False:
        i = 10
        return i + 15
    'Flip array in the left/right direction.\n\n    Args:\n        a (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return FlipLR().apply((a,))[0]