import numpy
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class Log1p(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            print('Hello World!')
        return 'log1p'

    def check_type_forward(self, in_types):
        if False:
            return 10
        type_check._argname(in_types, ('x',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, x):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        return (utils.force_array(numpy.log1p(x[0])),)

    def forward_gpu(self, x):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        return (cuda.cupy.log1p(x[0]),)

    def backward(self, indexes, gy):
        if False:
            return 10
        x = self.get_retained_inputs()
        return (gy[0] / (x[0] + 1.0),)

def log1p(x):
    if False:
        for i in range(10):
            print('nop')
    'Elementwise natural logarithm plus one function.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Log1p().apply((x,))[0]