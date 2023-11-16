import numpy
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check

class HardSigmoid(function_node.FunctionNode):
    """Hard-sigmoid function."""

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('x',))
        (x_type,) = in_types
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_cpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        x = inputs[0]
        y = numpy.clip(x * 0.2 + 0.5, 0.0, 1.0)
        self.retain_inputs((0,))
        return (utils.force_array(y, x.dtype),)

    def forward_gpu(self, inputs):
        if False:
            return 10
        x = inputs[0]
        self.retain_inputs((0,))
        return (cuda.elementwise('T x', 'T y', 'y = min(1.0, max(0.0, x * 0.2 + 0.5))', 'hard_sigmoid_fwd')(x),)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        (x,) = self.get_retained_inputs()
        return HardSigmoidGrad(x.data).apply(grad_outputs)

class HardSigmoidGrad(function_node.FunctionNode):
    """Hard-sigmoid gradient function."""

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.x = x

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        type_check._argname(in_types, ('gy',))
        type_check.expect(in_types[0].dtype.kind == 'f', in_types[0].dtype == self.x.dtype)

    def forward_cpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (gy,) = inputs
        gx = ((-2.5 < self.x) & (self.x < 2.5)) * gy * 0.2
        return (utils.force_array(gx, self.x.dtype),)

    def forward_gpu(self, inputs):
        if False:
            return 10
        (gy,) = inputs
        return (cuda.elementwise('T x, T g', 'T gx', 'gx = fabs(x) < 2.5 ? 0.2 * g : 0', 'hard_sigmoid_bwd')(self.x, gy),)

    def backward(self, indexes, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        return HardSigmoidGrad(self.x).apply(grad_outputs)

def hard_sigmoid(x):
    if False:
        for i in range(10):
            print('nop')
    'Element-wise hard-sigmoid function.\n\n    This function is defined as\n\n    .. math::\n\n        f(x) = \\left \\{ \\begin{array}{ll}\n        0 & {\\rm if}~ x < -2.5 \\\\\n        0.2 x + 0.5 & {\\rm if}~ -2.5 < x < 2.5 \\\\\n        1 & {\\rm if}~ 2.5 < x.\n        \\end{array} \\right.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable. A :math:`(s_1, s_2, ..., s_N)`-shaped float array.\n\n    Returns:\n        ~chainer.Variable: Output variable. A\n        :math:`(s_1, s_2, ..., s_N)`-shaped float array.\n\n    .. admonition:: Example\n\n        It maps the input values into the range of :math:`[0, 1]`.\n\n        >>> x = np.array([-2.6, -1, 0, 1, 2.6])\n        >>> x\n        array([-2.6, -1. ,  0. ,  1. ,  2.6])\n        >>> F.hard_sigmoid(x).array\n        array([0. , 0.3, 0.5, 0.7, 1. ])\n\n    '
    return HardSigmoid().apply((x,))[0]