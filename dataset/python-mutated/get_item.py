import numpy
import chainer
from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
from chainer import variable
import chainerx
_numpy_supports_0d_bool_index = numpy.lib.NumpyVersion(numpy.__version__) >= '1.13.0'

class GetItem(function_node.FunctionNode):
    """Function that slices array and extract elements."""

    def __init__(self, slices):
        if False:
            print('Hello World!')
        if isinstance(slices, list):
            if all([isinstance(s, int) for s in slices]):
                slices = (slices,)
            slices = tuple(slices)
        elif not isinstance(slices, tuple):
            slices = (slices,)
        if chainer.is_debug():
            n_ellipses = 0
            for s in slices:
                if s is Ellipsis:
                    n_ellipses += 1
            if n_ellipses > 1:
                raise ValueError('Only one Ellipsis is allowed')
        self.slices = slices

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check._argname(in_types, ('x',))

    def forward(self, xs):
        if False:
            while True:
                i = 10
        slices = tuple([backend.from_chx(s) if isinstance(s, chainerx.ndarray) else s for s in self.slices])
        return (utils.force_array(xs[0][slices]),)

    def backward(self, indexes, gy):
        if False:
            print('Hello World!')
        return GetItemGrad(self.slices, self.inputs[0].shape).apply(gy)

class GetItemGrad(function_node.FunctionNode):

    def __init__(self, slices, in_shape):
        if False:
            for i in range(10):
                print('nop')
        self.slices = slices
        self._in_shape = in_shape

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        slices = tuple([backend.from_chx(s) if isinstance(s, chainerx.ndarray) else s for s in self.slices])
        (gy,) = inputs
        xp = backend.get_array_module(*inputs)
        gx = xp.zeros(self._in_shape, gy.dtype)
        if xp is numpy:
            try:
                numpy.add.at(gx, slices, gy)
            except IndexError:
                done = False
                if not _numpy_supports_0d_bool_index and len(slices) == 1:
                    idx = numpy.asanyarray(slices[0])
                    if idx.dtype == numpy.dtype(bool):
                        numpy.add.at(gx[None], idx[None], gy)
                        done = True
                if not done:
                    msg = '\nGetItem does not support backward for this slices. The slices argument is not\nsupported by numpy.add.at, while it is supported by numpy.ndarray.__getitem__.\n\nPlease report this error to the issue tracker with the stack trace,\nthe information of your environment, and your script:\nhttps://github.com/chainer/chainer/issues/new.\n'
                    raise IndexError(msg)
        else:
            gx.scatter_add(slices, inputs[0])
        return (gx,)

    def backward(self, indexes, ggx):
        if False:
            print('Hello World!')
        return GetItem(self.slices).apply(ggx)

def get_item(x, slices):
    if False:
        print('Hello World!')
    "Extract elements from array with specified shape, axes and offsets.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            A variable to be sliced.\n        slices (int, slice, Ellipsis, None, integer array-like, boolean        array-like or tuple of them):\n            An object to specify the selection of elements.\n\n    Returns:\n        A :class:`~chainer.Variable` object which contains sliced array of\n        ``x``.\n\n    .. note::\n\n        It only supports types that are supported by CUDA's atomicAdd when\n        an integer array is included in ``slices``.\n        The supported types are ``numpy.float32``, ``numpy.int32``,\n        ``numpy.uint32``, ``numpy.uint64`` and ``numpy.ulonglong``.\n\n    .. note::\n\n        It does not support ``slices`` that contains multiple boolean arrays.\n\n    .. note::\n\n       See NumPy documentation for details of `indexing\n       <https://docs.scipy.org/doc/numpy/reference/arrays.indexing.html>`_.\n\n    .. admonition:: Example\n\n        >>> x = np.arange(12).reshape((2, 2, 3))\n        >>> x\n        array([[[ 0,  1,  2],\n                [ 3,  4,  5]],\n        <BLANKLINE>\n               [[ 6,  7,  8],\n                [ 9, 10, 11]]])\n        >>> F.get_item(x, 0)\n        variable([[0, 1, 2],\n                  [3, 4, 5]])\n        >>> F.get_item(x, (0, 0, slice(0, 2, 1)))  # equals x[0, 0, 0:2:1]\n        variable([0, 1])\n        >>> F.get_item(x, (Ellipsis, 2))  # equals x[..., 2]\n        variable([[ 2,  5],\n                  [ 8, 11]])\n        >>> F.get_item(x, (1, np.newaxis, 1, 0))  # equals x[1, None, 1, 0]\n        variable([9])\n\n    "
    return GetItem(slices).apply((x,))[0]

def install_variable_get_item():
    if False:
        print('Hello World!')
    variable.Variable.__getitem__ = get_item