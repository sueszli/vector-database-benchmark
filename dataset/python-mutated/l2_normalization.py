import six
from chainer import backend
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check

class _SetItemZero(function_node.FunctionNode):
    """Write values to mask of zero-initialized array"""

    def __init__(self, mask):
        if False:
            while True:
                i = 10
        self.mask = mask

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        xp = backend.get_array_module(x)
        y = xp.zeros(self.mask.shape, x.dtype)
        y[self.mask] = x
        return (y,)

    def backward(self, indices, grad_outputs):
        if False:
            while True:
                i = 10
        (g,) = grad_outputs
        return (g[self.mask],)

class NormalizeL2(function_node.FunctionNode):
    """L2 normalization"""

    def __init__(self, eps=1e-05, axis=1):
        if False:
            return 10
        self.eps = eps
        if isinstance(axis, six.integer_types):
            axis = (axis,)
        self.axis = axis

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check.expect(in_types.size() == 1)
        (x_type,) = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        self.retain_inputs((0,))
        (x,) = inputs
        xp = backend.get_array_module(x)
        norm = xp.sqrt(xp.sum(xp.square(x), axis=self.axis, keepdims=True), dtype=x.dtype) + x.dtype.type(self.eps)
        return (utils.force_array(x / norm),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (x,) = self.get_retained_inputs()
        (gy,) = grad_outputs
        F = chainer.functions
        norm_noeps = F.sqrt(F.sum(F.square(x), axis=self.axis, keepdims=True))
        norm = norm_noeps + self.eps
        x_gy_reduced = F.sum(x * gy, axis=self.axis, keepdims=True)
        mask = norm_noeps.array != 0
        (x_gy_reduced,) = _SetItemZero(mask).apply((x_gy_reduced[mask] / norm_noeps[mask],))
        gx = gy * norm - x_gy_reduced * x
        gx = gx / norm ** 2
        return (gx,)

def normalize(x, eps=1e-05, axis=1):
    if False:
        print('Hello World!')
    'Normalize input by L2 norm.\n\n    This function implements L2 normalization on a sample along the given\n    axis/axes. No reduction is done along the normalization axis.\n\n    In the case when :obj:`axis=1` and :math:`\\mathbf{x}` is a matrix of\n    dimension :math:`(N, K)`, where :math:`N` and :math:`K` denote mini-batch\n    size and the dimension of the input vectors, this function computes an\n    output matrix :math:`\\mathbf{y}` of dimension :math:`(N, K)` by the\n    following equation:\n\n    .. math::\n       \\mathbf{y}_i =\n           {\\mathbf{x}_i \\over \\| \\mathbf{x}_i \\|_2 + \\epsilon}\n\n    :obj:`eps` is used to avoid division by zero when norm of\n    :math:`\\mathbf{x}` along the given axis is zero.\n\n    The default value of :obj:`axis` is determined for backward compatibility.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            multi-dimensional output variable. The first\n            dimension is assumed to be the mini-batch dimension.\n        eps (float): Epsilon value for numerical stability.\n        axis (int or tuple of ints): Axis along which to normalize.\n\n    Returns:\n        ~chainer.Variable: The output variable which has the same shape\n        as :math:`x`.\n\n    '
    return NormalizeL2(eps, axis).apply((x,))[0]