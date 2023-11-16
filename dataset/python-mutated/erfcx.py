import numpy
try:
    from scipy import special
    available_cpu = True
except ImportError as e:
    available_cpu = False
    _import_error = e
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class Erfcx(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            while True:
                i = 10
        return 'erfcx'

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check.expect(in_types.size() == 1)
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            i = 10
            return i + 15
        if not available_cpu:
            raise ImportError('SciPy is not available. Forward computation of erfcx in CPU cannot be done. ' + str(_import_error))
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return (utils.force_array(special.erfcx(x[0]), dtype=x[0].dtype),)

    def forward_gpu(self, x):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        self.retain_outputs((0,))
        return (cuda.elementwise('T x', 'T y', 'y = erfcx(x)', 'elementwise_erfcx')(x[0]),)

    def backward(self, indexes, gy):
        if False:
            print('Hello World!')
        x = self.get_retained_inputs()[0]
        y = self.get_retained_outputs()[0]
        return (2 * (x * y - numpy.pi ** (-0.5)) * gy[0],)

def erfcx(x):
    if False:
        for i in range(10):
            print('nop')
    'Elementwise scaled complementary error function.\n\n    .. note::\n       Forward computation in CPU cannot be done if\n       `SciPy <https://www.scipy.org/>`_ is not available.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Erfcx().apply((x,))[0]