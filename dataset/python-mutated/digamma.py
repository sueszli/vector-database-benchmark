import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
_digamma_cpu = None

class DiGamma(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        return 'digamma'

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            i = 10
            return i + 15
        global _digamma_cpu
        if _digamma_cpu is None:
            try:
                from scipy import special
                _digamma_cpu = special.digamma
            except ImportError:
                raise ImportError('SciPy is not available. Forward computation of digamma can not be done.')
        self.retain_inputs((0,))
        return (utils.force_array(_digamma_cpu(x[0]), dtype=x[0].dtype),)

    def forward_gpu(self, x):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        return (utils.force_array(cuda.cupyx.scipy.special.digamma(x[0]), dtype=x[0].dtype),)

    def backward(self, indexes, gy):
        if False:
            for i in range(10):
                print('nop')
        z = self.get_retained_inputs()[0]
        xp = backend.get_array_module(*gy)
        return (chainer.functions.polygamma(xp.array(1), z) * gy[0],)

def digamma(x):
    if False:
        return 10
    'Digamma function.\n\n    .. note::\n       Forward computation in CPU can not be done if\n       `SciPy <https://www.scipy.org/>`_ is not available.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return DiGamma().apply((x,))[0]