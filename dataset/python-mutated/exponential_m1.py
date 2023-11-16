import numpy
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class Expm1(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            for i in range(10):
                print('nop')
        return 'expm1'

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            while True:
                i = 10
        self.retain_outputs((0,))
        return (utils.force_array(numpy.expm1(x[0])),)

    def forward_gpu(self, x):
        if False:
            print('Hello World!')
        self.retain_outputs((0,))
        return (cuda.cupy.expm1(x[0]),)

    def backward(self, indexes, gy):
        if False:
            for i in range(10):
                print('nop')
        y = self.get_retained_outputs()[0]
        return ((y + 1.0) * gy[0],)

def expm1(x):
    if False:
        for i in range(10):
            print('nop')
    'Elementwise exponential minus one function.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Expm1().apply((x,))[0]