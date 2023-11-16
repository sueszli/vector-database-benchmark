from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
_zeta_cpu = None

class Zeta(function_node.FunctionNode):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self._x = x

    @property
    def label(self):
        if False:
            return 10
        return 'zeta'

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check._argname(in_types, 'q')
        (q_type,) = in_types
        type_check.expect(q_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        if False:
            print('Hello World!')
        (q,) = inputs
        global _zeta_cpu
        if _zeta_cpu is None:
            try:
                from scipy import special
                _zeta_cpu = special.zeta
            except ImportError:
                raise ImportError('Scipy is not available. Forward computation of zeta cannot be done.')
        self.retain_inputs((0,))
        return (utils.force_array(_zeta_cpu(self._x, q), dtype=q.dtype),)

    def forward_gpu(self, inputs):
        if False:
            i = 10
            return i + 15
        (q,) = inputs
        self.retain_inputs((0,))
        return (utils.force_array(cuda.cupyx.scipy.special.zeta(self._x, q), dtype=q.dtype),)

    def backward(self, indexes, gy):
        if False:
            print('Hello World!')
        (q,) = self.get_retained_inputs()
        return (gy[0] * -self._x * zeta(self._x + 1, q),)

def zeta(x, q):
    if False:
        return 10
    'Zeta function.\n\n    Differentiable only with respect to q\n\n    .. note::\n       Forward computation in CPU can not be done if\n       `SciPy <https://www.scipy.org/>`_ is not available.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n        q (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Zeta(x).apply((q,))[0]