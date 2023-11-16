from chainer import function_node
from chainer.utils import type_check

class Swapaxes(function_node.FunctionNode):
    """Swap two axes of an array."""

    def __init__(self, axis1, axis2):
        if False:
            return 10
        self.axis1 = axis1
        self.axis2 = axis2

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check.expect(in_types.size() == 1)

    @property
    def label(self):
        if False:
            while True:
                i = 10
        return 'Swapaxes'

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        (x,) = inputs
        return (x.swapaxes(self.axis1, self.axis2),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        (gy,) = grad_outputs
        return Swapaxes(self.axis1, self.axis2).apply((gy,))

def swapaxes(x, axis1, axis2):
    if False:
        print('Hello World!')
    'Swap two axes of a variable.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Input variable.\n            A :math:`(s_1, s_2, ..., s_N)` -shaped float array.\n        axis1 (int): The first axis to swap.\n        axis2 (int): The second axis to swap.\n\n    Returns:\n        ~chainer.Variable: Variable whose axes are swapped.\n\n    .. admonition:: Example\n\n        >>> x = np.array([[[0, 1, 2], [3, 4, 5]]], np.float32)\n        >>> x.shape\n        (1, 2, 3)\n        >>> y = F.swapaxes(x, axis1=0, axis2=1)\n        >>> y.shape\n        (2, 1, 3)\n        >>> y.array\n        array([[[0., 1., 2.]],\n        <BLANKLINE>\n               [[3., 4., 5.]]], dtype=float32)\n\n    '
    (y,) = Swapaxes(axis1, axis2).apply((x,))
    return y