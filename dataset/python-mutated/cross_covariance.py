import chainer
from chainer import backend
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class CrossCovariance(function_node.FunctionNode):
    """Cross-covariance loss."""

    def __init__(self, reduce='half_squared_sum'):
        if False:
            print('Hello World!')
        self.y_centered = None
        self.z_centered = None
        self.covariance = None
        if reduce not in ('half_squared_sum', 'no'):
            raise ValueError("Only 'half_squared_sum' and 'no' are valid for 'reduce', but '%s' is given" % reduce)
        self.reduce = reduce

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check._argname(in_types, ('y', 'z'))
        (y_type, z_type) = in_types
        type_check.expect(y_type.dtype.kind == 'f', y_type.dtype == z_type.dtype, y_type.ndim == 2, z_type.ndim == 2, y_type.shape[0] == z_type.shape[0])

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        (y, z) = inputs
        self.retain_inputs((0, 1))
        y_centered = y - y.mean(axis=0, keepdims=True)
        z_centered = z - z.mean(axis=0, keepdims=True)
        covariance = y_centered.T.dot(z_centered)
        covariance /= len(y)
        if self.reduce == 'half_squared_sum':
            xp = backend.get_array_module(*inputs)
            cost = xp.vdot(covariance, covariance)
            cost *= y.dtype.type(0.5)
            return (utils.force_array(cost),)
        else:
            return (covariance,)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (y, z) = self.get_retained_inputs()
        (gcost,) = grad_outputs
        y_mean = chainer.functions.mean(y, axis=0, keepdims=True)
        z_mean = chainer.functions.mean(z, axis=0, keepdims=True)
        y_centered = y - chainer.functions.broadcast_to(y_mean, y.shape)
        z_centered = z - chainer.functions.broadcast_to(z_mean, z.shape)
        gcost_div_n = gcost / gcost.dtype.type(len(y))
        ret = []
        if self.reduce == 'half_squared_sum':
            covariance = chainer.functions.matmul(y_centered.T, z_centered)
            covariance /= len(y)
            if 0 in indexes:
                gy = chainer.functions.matmul(z_centered, covariance.T)
                gy *= chainer.functions.broadcast_to(gcost_div_n, gy.shape)
                ret.append(gy)
            if 1 in indexes:
                gz = chainer.functions.matmul(y_centered, covariance)
                gz *= chainer.functions.broadcast_to(gcost_div_n, gz.shape)
                ret.append(gz)
        else:
            if 0 in indexes:
                gy = chainer.functions.matmul(z_centered, gcost_div_n.T)
                ret.append(gy)
            if 1 in indexes:
                gz = chainer.functions.matmul(y_centered, gcost_div_n)
                ret.append(gz)
        return ret

def cross_covariance(y, z, reduce='half_squared_sum'):
    if False:
        i = 10
        return i + 15
    "Computes the sum-squared cross-covariance penalty between ``y`` and ``z``\n\n    The output is a variable whose value depends on the value of\n    the option ``reduce``. If it is ``'no'``, it holds the covariant\n    matrix that has as many rows (resp. columns) as the dimension of\n    ``y`` (resp.z).\n    If it is ``'half_squared_sum'``, it holds the half of the\n    Frobenius norm (i.e. L2 norm of a matrix flattened to a vector)\n    of the covarianct matrix.\n\n    Args:\n        y (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Variable holding a matrix where the first dimension\n            corresponds to the batches.\n        z (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Variable holding a matrix where the first dimension\n            corresponds to the batches.\n        reduce (str): Reduction option. Its value must be either\n            ``'half_squared_sum'`` or ``'no'``.\n            Otherwise, :class:`ValueError` is raised.\n\n    Returns:\n        ~chainer.Variable:\n            A variable holding the cross covariance loss.\n            If ``reduce`` is ``'no'``, the output variable holds\n            2-dimensional array matrix of shape ``(M, N)`` where\n            ``M`` (resp. ``N``) is the number of columns of ``y``\n            (resp. ``z``).\n            If it is ``'half_squared_sum'``, the output variable\n            holds a scalar value.\n\n    .. note::\n\n       This cost can be used to disentangle variables.\n       See https://arxiv.org/abs/1412.6583v3 for details.\n\n    "
    return CrossCovariance(reduce).apply((y, z))[0]