try:
    from scipy import special
    available_cpu = True
except ImportError as e:
    available_cpu = False
    _import_error = e
import math
import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
BACKWORDC = math.pi ** 0.5 / 2

class ErfcInv(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        return 'erfcinv'

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation of erfcinv in CPU cannot be done. ' + str(_import_error))
        self.retain_outputs((0,))
        return (utils.force_array(special.erfcinv(x[0]), dtype=x[0].dtype),)

    def forward_gpu(self, x):
        if False:
            return 10
        self.retain_outputs((0,))
        return (cuda.elementwise('T x', 'T y', 'y = erfcinv(x)', 'elementwise_erfcinv')(x[0]),)

    def backward(self, indexes, gy):
        if False:
            for i in range(10):
                print('nop')
        (y,) = self.get_retained_outputs()
        return (-BACKWORDC * chainer.functions.exp(y ** 2) * gy[0],)

def erfcinv(x):
    if False:
        return 10
    'Elementwise inverse function of complementary error function.\n\n    .. note::\n       Forward computation in CPU cannot be done if\n       `SciPy <https://www.scipy.org/>`_ is not available.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return ErfcInv().apply((x,))[0]