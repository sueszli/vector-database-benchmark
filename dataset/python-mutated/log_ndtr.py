import numpy
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.math import erfcx
from chainer import utils
from chainer.utils import type_check
_log_ndtr_cpu = None

class LogNdtr(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        return 'log_ndtr'

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            while True:
                i = 10
        global _log_ndtr_cpu
        if _log_ndtr_cpu is None:
            try:
                from scipy import special
                _log_ndtr_cpu = special.log_ndtr
            except ImportError:
                raise ImportError('SciPy is not available. Forward computation of log_ndtr can not be done.')
        self.retain_inputs((0,))
        return (utils.force_array(_log_ndtr_cpu(x[0]), dtype=x[0].dtype),)

    def forward_gpu(self, x):
        if False:
            print('Hello World!')
        self.retain_inputs((0,))
        return (cuda.elementwise('T x', 'T y', '\n            if (x > 0) {\n                y = log1p(-normcdf(-x));\n            } else {\n                y = log(0.5 * erfcx(-sqrt(0.5) * x)) - 0.5 * x * x;\n            }\n            ', 'elementwise_log_ndtr')(x[0]),)

    def backward(self, indexes, gy):
        if False:
            for i in range(10):
                print('nop')
        x = self.get_retained_inputs()[0]
        return ((2 / numpy.pi) ** 0.5 / erfcx.erfcx(-x / 2 ** 0.5) * gy[0],)

def log_ndtr(x):
    if False:
        return 10
    'Logarithm of cumulative distribution function of normal distribution.\n\n    .. note::\n       Forward computation in CPU can not be done if\n       `SciPy <https://www.scipy.org/>`_ is not available.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return LogNdtr().apply((x,))[0]