import six
from chainer import backend
from chainer import function_node
from chainer.utils import type_check

class Rollaxis(function_node.FunctionNode):
    """Roll axis of an array."""

    def __init__(self, axis, start):
        if False:
            for i in range(10):
                print('nop')
        if not isinstance(axis, six.integer_types):
            raise TypeError('axis must be int')
        if not isinstance(start, six.integer_types):
            raise TypeError('start must be int')
        self.axis = axis
        self.start = start

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        if self.axis >= 0:
            type_check.expect(x_type.ndim > self.axis)
        else:
            type_check.expect(x_type.ndim > -self.axis - 1)
        if self.start >= 0:
            type_check.expect(x_type.ndim >= self.start)
        else:
            type_check.expect(x_type.ndim > -self.start - 1)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        self.retain_inputs(())
        self._in_ndim = inputs[0].ndim
        xp = backend.get_array_module(*inputs)
        return (xp.rollaxis(inputs[0], self.axis, self.start),)

    def backward(self, indexes, gy):
        if False:
            while True:
                i = 10
        axis = self.axis
        if axis < 0:
            axis += self._in_ndim
        start = self.start
        if start < 0:
            start += self._in_ndim
        if axis > start:
            axis += 1
        elif axis < start:
            start -= 1
        return Rollaxis(start, axis).apply(gy)

def rollaxis(x, axis, start=0):
    if False:
        i = 10
        return i + 15
    'Roll the axis backwards to the given position.\n\n    This function continues to be supported for backward compatibility,\n    but you should prefer\n    ``chainer.functions.moveaxis(x, source, destination)``.\n    See :func:`chainer.functions.moveaxis`.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n        axis (int): The axis to roll backwards.\n        start (int): The place to which the axis is moved.\n\n    Returns:\n        ~chainer.Variable: Variable whose axis is rolled.\n    '
    return Rollaxis(axis, start).apply((x,))[0]