import math
import warnings
import numpy
import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.math import exponential
from chainer import utils
from chainer.utils import type_check
_ndtr_cpu = None

def _slow_ndtr_cpu(x):
    if False:
        print('Hello World!')
    return 0.5 * math.erfc(-x / 2 ** 0.5)

class Ndtr(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            print('Hello World!')
        return 'ndtr'

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            print('Hello World!')
        global _ndtr_cpu
        if _ndtr_cpu is None:
            try:
                from scipy import special
                _ndtr_cpu = special.ndtr
            except ImportError:
                warnings.warn('SciPy is not available. Forward computation of ndtr in CPU can be slow without SciPy.', chainer.warnings.PerformanceWarning)
                _ndtr_cpu = numpy.vectorize(_slow_ndtr_cpu)
        self.retain_inputs((0,))
        return (utils.force_array(_ndtr_cpu(x[0]), dtype=x[0].dtype),)

    def forward_gpu(self, x):
        if False:
            return 10
        self.retain_inputs((0,))
        return (cuda.elementwise('T x', 'T y', 'y = normcdf(x)', 'elementwise_ndtr')(x[0]),)

    def backward(self, indexes, gy):
        if False:
            i = 10
            return i + 15
        x = self.get_retained_inputs()[0]
        return ((2 * numpy.pi) ** (-0.5) * exponential.exp(-0.5 * x ** 2) * gy[0],)

def ndtr(x):
    if False:
        return 10
    'Elementwise cumulative distribution function of normal distribution.\n\n    .. note::\n       Forward computation in CPU can be slow if\n       `SciPy <https://www.scipy.org/>`_ is not available.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Ndtr().apply((x,))[0]