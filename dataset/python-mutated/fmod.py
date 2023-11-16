from chainer import backend
from chainer import function_node
import chainer.functions
from chainer import utils
from chainer.utils import type_check

class Fmod(function_node.FunctionNode):

    @property
    def label(self):
        if False:
            print('Hello World!')
        return 'fmod'

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('x', 'divisor'))
        type_check.expect(in_types[0].dtype == in_types[1].dtype, in_types[0].dtype.kind == 'f', in_types[1].dtype.kind == 'f')

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(*inputs)
        (x, divisor) = inputs
        m = xp.fmod(x, divisor)
        return (utils.force_array(m, x.dtype),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (x, divisor) = self.get_retained_inputs()
        (gw,) = grad_outputs
        return (gw, -chainer.functions.fix(x / divisor) * gw)

def fmod(x, divisor):
    if False:
        for i in range(10):
            print('nop')
    'Elementwise mod function.\n\n    .. math::\n       y_i = x_i \\bmod \\mathrm{divisor}.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n        divisor (:class:`~chainer.Variable` or :ref:`ndarray`): Input divisor.\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Fmod().apply((x, divisor))[0]