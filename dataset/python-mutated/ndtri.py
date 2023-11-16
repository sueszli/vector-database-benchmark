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

class Ndtri(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        return 'ndtri'

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            print('Hello World!')
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation of ndtri in CPU can not be done.' + str(_import_error))
        self.retain_outputs((0,))
        return (utils.force_array(special.ndtri(x[0]), dtype=x[0].dtype),)

    def forward_gpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.retain_outputs((0,))
        return (cuda.elementwise('T x', 'T y', 'y = normcdfinv(x)', 'elementwise_ndtri')(x[0]),)

    def backward(self, indexes, gy):
        if False:
            return 10
        (y,) = self.get_retained_outputs()
        sqrt_2pi = (2 * math.pi) ** 0.5
        return (sqrt_2pi * chainer.functions.exp(0.5 * y ** 2) * gy[0],)

def ndtri(x):
    if False:
        i = 10
        return i + 15
    'Elementwise inverse function of ndtr.\n\n    .. note::\n       Forward computation in CPU can not be done if\n       `SciPy <https://www.scipy.org/>`_ is not available.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Ndtri().apply((x,))[0]