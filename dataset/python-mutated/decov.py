from chainer import backend
from chainer import function
from chainer import utils
from chainer.utils import type_check

class DeCov(function.Function):
    """DeCov loss (https://arxiv.org/abs/1511.06068)"""

    def __init__(self, reduce='half_squared_sum'):
        if False:
            for i in range(10):
                print('nop')
        self.h_centered = None
        self.covariance = None
        if reduce not in ('half_squared_sum', 'no'):
            raise ValueError("only 'half_squared_sum' and 'no' are valid for 'reduce', but '%s' is given" % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check._argname(in_types, ('h',))
        (h_type,) = in_types
        type_check.expect(h_type.dtype.kind == 'f', h_type.ndim == 2)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        xp = backend.get_array_module(*inputs)
        (h,) = inputs
        self.h_centered = h - h.mean(axis=0, keepdims=True)
        self.covariance = self.h_centered.T.dot(self.h_centered)
        xp.fill_diagonal(self.covariance, 0.0)
        self.covariance /= len(h)
        if self.reduce == 'half_squared_sum':
            cost = xp.vdot(self.covariance, self.covariance)
            cost *= h.dtype.type(0.5)
            return (utils.force_array(cost),)
        else:
            return (self.covariance,)

    def backward(self, inputs, grad_outputs):
        if False:
            while True:
                i = 10
        xp = backend.get_array_module(*inputs)
        (h,) = inputs
        (gcost,) = grad_outputs
        gcost_div_n = gcost / gcost.dtype.type(len(h))
        if self.reduce == 'half_squared_sum':
            gh = 2.0 * self.h_centered.dot(self.covariance)
            gh *= gcost_div_n
        else:
            xp.fill_diagonal(gcost_div_n, 0.0)
            gh = self.h_centered.dot(gcost_div_n + gcost_div_n.T)
        return (gh,)

def decov(h, reduce='half_squared_sum'):
    if False:
        while True:
            i = 10
    "Computes the DeCov loss of ``h``\n\n    The output is a variable whose value depends on the value of\n    the option ``reduce``. If it is ``'no'``, it holds a matrix\n    whose size is same as the number of columns of ``y``.\n    If it is ``'half_squared_sum'``, it holds the half of the\n    squared Frobenius norm (i.e. squared of the L2 norm of a matrix flattened\n    to a vector) of the matrix.\n\n    Args:\n        h (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Variable holding a matrix where the first dimension\n            corresponds to the batches.\n        reduce (str): Reduction option. Its value must be either\n            ``'half_squared_sum'`` or ``'no'``.\n            Otherwise, :class:`ValueError` is raised.\n\n    Returns:\n        ~chainer.Variable:\n            A variable holding a scalar of the DeCov loss.\n            If ``reduce`` is ``'no'``, the output variable holds\n            2-dimensional array matrix of shape ``(N, N)`` where\n            ``N`` is the number of columns of ``y``.\n            If it is ``'half_squared_sum'``, the output variable\n            holds a scalar value.\n\n    .. note::\n\n       See https://arxiv.org/abs/1511.06068 for details.\n\n    "
    return DeCov(reduce)(h)