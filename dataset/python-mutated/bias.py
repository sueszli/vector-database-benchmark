import chainer
from chainer.functions.array import broadcast
from chainer.functions.array import reshape

def bias(x, y, axis=1):
    if False:
        for i in range(10):
            print('nop')
    'Elementwise summation with broadcasting.\n\n    Computes a elementwise summation of two input variables, with the shape of\n    the latter variable broadcasted to match the shape of the former. ``axis``\n    is the first axis of the first variable along which the second variable is\n    applied.\n\n    The term "broadcasting" here comes from Caffe\'s bias layer so the\n    "broadcasting" with the following arguments::\n\n           x : 100 x 3 x 40 x 5 x 6\n           y : 3 x 40\n        axis : 1\n\n    is equivalent to the following numpy broadcasting::\n\n        x : 100 x  3 x 40 x 5 x 6\n        y :  (1 x) 3 x 40 x 1 x 1\n\n    Note that the axis of ``x`` to which we apply ``y`` is specified by the\n    argument ``axis``, whose meaning is different from numpy\'s ``axis``.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable to be summed.\n        y (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable to sum, broadcasted.\n        axis (int): The first axis of ``x`` along which ``y`` is applied.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    '
    x_shape = x.shape
    y_shape = y.shape
    if chainer.is_debug():
        assert x_shape[axis:axis + len(y_shape)] == y_shape
    y1_shape = tuple([1] * axis + list(y_shape) + [1] * (len(x_shape) - axis - len(y_shape)))
    y1 = reshape.reshape(y, y1_shape)
    y2 = broadcast.broadcast_to(y1, x_shape)
    return x + y2