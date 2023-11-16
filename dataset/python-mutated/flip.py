import six
from chainer import backend
from chainer import function_node
from chainer.utils import type_check

def _flip(array, axis):
    if False:
        i = 10
        return i + 15
    indices = [slice(None)] * array.ndim
    indices[axis] = slice(None, None, -1)
    return array[tuple(indices)]

class Flip(function_node.FunctionNode):
    """Flips an input variable in reverse order along the given axis."""

    def __init__(self, axis):
        if False:
            i = 10
            return i + 15
        if not isinstance(axis, six.integer_types):
            raise TypeError('axis must be int')
        self.axis = axis

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        type_check.expect(x_type.ndim > 0)
        if self.axis >= 0:
            type_check.expect(x_type.ndim > self.axis)
        else:
            type_check.expect(x_type.ndim >= -self.axis)

    def forward(self, inputs):
        if False:
            return 10
        xp = backend.get_array_module(*inputs)
        if hasattr(xp, 'flip'):
            return (xp.flip(inputs[0], self.axis),)
        else:
            return (_flip(inputs[0], self.axis),)

    def backward(self, indexes, grad_outputs):
        if False:
            i = 10
            return i + 15
        return (flip(grad_outputs[0], self.axis),)

def flip(x, axis):
    if False:
        while True:
            i = 10
    'Flips an input variable in reverse order along the given axis.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable.\n        axis (int): Axis along which the input variable is reversed.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return Flip(axis).apply((x,))[0]