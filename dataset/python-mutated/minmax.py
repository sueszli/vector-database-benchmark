import numpy
import six
from chainer import backend
from chainer import function_node
import chainer.functions
import chainer.utils
from chainer.utils import type_check
import chainerx

class SelectorBase(function_node.FunctionNode):
    """Select an array element from a given axis or set of axes."""

    def __init__(self, axis=None, keepdims=False):
        if False:
            print('Hello World!')
        self.keepdims = keepdims
        if axis is None:
            self.axis = None
        elif isinstance(axis, six.integer_types):
            self.axis = (axis,)
        elif isinstance(axis, tuple) and all((isinstance(a, six.integer_types) for a in axis)):
            if len(set(axis)) != len(axis):
                raise ValueError('duplicate value in axis: ({})'.format(', '.join(map(str, axis))))
            self.axis = axis
        else:
            raise TypeError('None, int or tuple of int are required')

    def _fwd(self, x, xp):
        if False:
            while True:
                i = 10
        raise NotImplementedError('_fwd should be implemented in sub-class.')

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')
        if self.axis is not None:
            for axis in self.axis:
                if axis >= 0:
                    type_check.expect(axis < in_types[0].ndim)
                else:
                    type_check.expect(-axis - 1 < in_types[0].ndim)

    def forward(self, x):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        xp = backend.get_array_module(*x)
        return (xp.asarray(self._fwd(x[0], xp)),)

    def backward(self, indexes, gy):
        if False:
            print('Hello World!')
        x = self.get_retained_inputs()[0]
        y = self.get_retained_outputs()[0]
        if self.axis is None:
            axis = range(x.ndim)
        else:
            axis = [ax % x.ndim for ax in self.axis]
        shape = [s if ax not in axis else 1 for (ax, s) in enumerate(x.shape)]
        gy = gy[0].reshape(shape)
        y = y.reshape(shape)
        cond = x.data == y.data
        gy = chainer.functions.broadcast_to(gy, cond.shape)
        return (gy * cond,)

class Max(SelectorBase):

    def forward_chainerx(self, x):
        if False:
            for i in range(10):
                print('nop')
        return (chainerx.amax(x[0], axis=self.axis, keepdims=self.keepdims),)

    def _fwd(self, x, xp):
        if False:
            i = 10
            return i + 15
        return xp.amax(x, axis=self.axis, keepdims=self.keepdims)

class Min(SelectorBase):

    def forward_chainerx(self, x):
        if False:
            return 10
        return (chainerx.amin(x[0], axis=self.axis, keepdims=self.keepdims),)

    def _fwd(self, x, xp):
        if False:
            i = 10
            return i + 15
        return xp.amin(x, axis=self.axis, keepdims=self.keepdims)

class IndexSelectorBase(function_node.FunctionNode):
    """Select index of an array element from a given axis."""

    def __init__(self, axis=None):
        if False:
            for i in range(10):
                print('nop')
        if axis is None:
            self.axis = None
        elif isinstance(axis, six.integer_types):
            self.axis = axis
        else:
            raise TypeError('None or int are required')

    def _fwd(self, x, xp):
        if False:
            i = 10
            return i + 15
        raise NotImplementedError('_fwd should be implemented in sub-class.')

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check.expect(in_types.size() == 1, in_types[0].dtype.kind == 'f')
        if self.axis is not None:
            if self.axis >= 0:
                type_check.expect(self.axis < in_types[0].ndim)
            else:
                type_check.expect(-self.axis - 1 < in_types[0].ndim)

    def forward(self, x):
        if False:
            while True:
                i = 10
        xp = backend.get_array_module(*x)
        return (xp.asarray(self._fwd(x[0], xp)),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        return (None,)

class ArgMin(IndexSelectorBase):

    def forward_chainerx(self, x):
        if False:
            print('Hello World!')
        return (chainerx.argmin(x[0], axis=self.axis).astype(numpy.int32),)

    def _fwd(self, x, xp):
        if False:
            for i in range(10):
                print('nop')
        return xp.argmin(x, axis=self.axis).astype(numpy.int32)

class ArgMax(IndexSelectorBase):

    def forward_chainerx(self, x):
        if False:
            for i in range(10):
                print('nop')
        return (chainerx.argmax(x[0], axis=self.axis).astype(numpy.int32),)

    def _fwd(self, x, xp):
        if False:
            i = 10
            return i + 15
        return xp.argmax(x, axis=self.axis).astype(numpy.int32)

def max(x, axis=None, keepdims=False):
    if False:
        while True:
            i = 10
    'Maximum of array elements over a given axis.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Array to be maximized.\n        axis (None, int, or tuple of int): Axis over which a max is performed.\n            The default (axis = None) is perform a max over all the dimensions\n            of the input array.\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return Max(axis, keepdims).apply((x,))[0]

def min(x, axis=None, keepdims=False):
    if False:
        while True:
            i = 10
    'Minimum of array elements over a given axis.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Array to be minimized.\n        axis (None, int, or tuple of int): Axis over which a min is performed.\n            The default (axis = None) is perform a min over all the dimensions\n            of the input array.\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return Min(axis, keepdims).apply((x,))[0]

def argmax(x, axis=None):
    if False:
        i = 10
        return i + 15
    'Returns index which holds maximum of array elements over a given axis.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Array to find maximum elements.\n        axis (None or int): Axis over which a max is performed.\n            The default (axis = None) is perform a max over all the dimensions\n            of the input array.\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return ArgMax(axis).apply((x,))[0]

def argmin(x, axis=None):
    if False:
        for i in range(10):
            print('nop')
    'Returns index which holds minimum of array elements over a given axis.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Array to find minimum elements.\n        axis (None or int): Axis over which a min is performed.\n            The default (axis = None) is perform a min over all the dimensions\n            of the input array.\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    return ArgMin(axis).apply((x,))[0]