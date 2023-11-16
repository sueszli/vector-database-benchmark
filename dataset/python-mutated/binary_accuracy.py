from __future__ import division
from chainer import backend
from chainer import function
from chainer.utils import type_check

class BinaryAccuracy(function.Function):
    ignore_label = -1

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check._argname(in_types, ('x', 't'))
        (x_type, t_type) = in_types
        type_check.expect(x_type.dtype.kind == 'f', t_type.dtype.kind == 'i', t_type.shape == x_type.shape)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        xp = backend.get_array_module(*inputs)
        (y, t) = inputs
        y = y.ravel()
        t = t.ravel()
        c = y >= 0
        count = xp.maximum(1, (t != self.ignore_label).sum())
        return (xp.asarray((c == t).sum() / count, dtype=y.dtype),)

def binary_accuracy(y, t):
    if False:
        for i in range(10):
            print('nop')
    'Computes binary classification accuracy of the minibatch.\n\n    Args:\n        y (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Array whose i-th element indicates the score of\n            positive at the i-th sample.\n            The prediction label :math:`\\hat t[i]` is ``1`` if\n            ``y[i] >= 0``, otherwise ``0``.\n\n        t (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Array holding a signed integer vector of ground truth labels.\n            If ``t[i] == 1``, it indicates that i-th sample is positive.\n            If ``t[i] == 0``, it indicates that i-th sample is negative.\n            If ``t[i] == -1``, corresponding ``y[i]`` is ignored.\n            Accuracy is zero if all ground truth labels are ``-1``.\n\n    Returns:\n        ~chainer.Variable: A variable holding a scalar array of the accuracy.\n\n    .. note:: This function is non-differentiable.\n\n    .. admonition:: Example\n\n        We show the most common case, when ``y`` is the two dimensional array.\n\n        >>> y = np.array([[-2.0, 0.0], # prediction labels are [0, 1]\n        ...               [3.0, -5.0]]) # prediction labels are [1, 0]\n        >>> t = np.array([[0, 1],\n        ...              [1, 0]], np.int32)\n        >>> F.binary_accuracy(y, t).array # 100% accuracy because all samples are correct.\n        array(1.)\n        >>> t = np.array([[0, 0],\n        ...              [1, 1]], np.int32)\n        >>> F.binary_accuracy(y, t).array # 50% accuracy because y[0][0] and y[1][0] are correct.\n        array(0.5)\n        >>> t = np.array([[0, -1],\n        ...              [1, -1]], np.int32)\n        >>> F.binary_accuracy(y, t).array # 100% accuracy because of ignoring y[0][1] and y[1][1].\n        array(1.)\n    '
    return BinaryAccuracy()(y, t)