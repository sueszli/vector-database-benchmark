import numpy
import chainer
from chainer.backends import cuda
from chainer import function_node
from chainer import utils
from chainer.utils import type_check
import chainerx
if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    _mode = cuda.cuda.cudnn.CUDNN_ACTIVATION_CLIPPED_RELU

class ClippedReLU(function_node.FunctionNode):
    """Clipped Rectifier Unit function.

    Clipped ReLU is written as
    :math:`ClippedReLU(x, z) = \\min(\\max(0, x), z)`,
    where :math:`z(>0)` is a parameter to cap return value of ReLU.

    """
    _use_cudnn = False

    def __init__(self, z):
        if False:
            print('Hello World!')
        if not isinstance(z, float):
            raise TypeError('z must be float value')
        assert z > 0
        self.cap = z

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f')

    def forward_chainerx(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (x,) = inputs
        return (chainerx.clipped_relu(x, self.cap),)

    def forward_cpu(self, inputs):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        (x,) = inputs
        return (utils.force_array(numpy.minimum(numpy.maximum(0, x), self.cap), x.dtype),)

    def forward_gpu(self, inputs):
        if False:
            print('Hello World!')
        self.retain_inputs((0,))
        (x,) = inputs
        if chainer.should_use_cudnn('==always') and x.flags.c_contiguous:
            self._use_cudnn = True
            y = cudnn.activation_forward(x, _mode, self.cap)
            self.retain_outputs((0,))
        else:
            return (cuda.elementwise('T x, T cap', 'T y', 'y = min(max(x, (T)0), cap)', 'clipped_relu_fwd')(x, self.cap),)
        return (y,)

    def backward(self, indexes, grad_outputs):
        if False:
            print('Hello World!')
        (x,) = self.get_retained_inputs()
        if chainer.should_use_cudnn('==always') and self._use_cudnn:
            y = self.get_retained_outputs()[0]
            return ClippedReLUGrad3(x.data, y.data, self.cap).apply(grad_outputs)
        else:
            return ClippedReLUGrad2(x.data, self.cap).apply(grad_outputs)

class ClippedReLUGrad2(function_node.FunctionNode):
    """Clipped Rectifier Unit gradient function."""

    def __init__(self, x, z):
        if False:
            for i in range(10):
                print('nop')
        self.x = x
        self.cap = z

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check._argname(in_types, ('gy',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        if False:
            while True:
                i = 10
        (gy,) = inputs
        x = self.x
        return (utils.force_array(gy * (0 < x) * (x < self.cap), x.dtype),)

    def forward_gpu(self, inputs):
        if False:
            print('Hello World!')
        (gy,) = inputs
        gx = cuda.elementwise('T x, T gy, T z', 'T gx', 'gx = ((x > 0) & (x < z)) ? gy : (T)0', 'clipped_relu_bwd')(self.x, gy, self.cap)
        return (gx,)

    def backward(self, indexes, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        return ClippedReLUGrad2(self.x, self.cap).apply(grad_outputs)

class ClippedReLUGrad3(function_node.FunctionNode):
    """Clipped Rectifier Unit gradient function."""

    def __init__(self, x, y, z):
        if False:
            while True:
                i = 10
        self.x = x
        self.y = y
        self.cap = z

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check._argname(in_types, ('gy',))
        type_check.expect(in_types[0].dtype.kind == 'f')

    def forward_cpu(self, inputs):
        if False:
            while True:
                i = 10
        (gy,) = inputs
        return (utils.force_array(gy * (0 < self.x) * (self.x < self.cap), self.x.dtype),)

    def forward_gpu(self, inputs):
        if False:
            return 10
        assert chainer.should_use_cudnn('==always')
        return (cudnn.activation_backward(self.x, self.y, inputs[0], _mode, self.cap),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        return ClippedReLUGrad3(self.x, self.y, self.cap).apply(grad_outputs)

def clipped_relu(x, z=20.0):
    if False:
        for i in range(10):
            print('nop')
    'Clipped Rectifier Unit function.\n\n    For a clipping value :math:`z(>0)`, it computes\n\n    .. math:: \\text{ClippedReLU}(x, z) = \\min(\\max(0, x), z).\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable. A :math:`(s_1, s_2, ..., s_n)`-shaped float array.\n        z (float): Clipping value. (default = 20.0)\n\n    Returns:\n        ~chainer.Variable: Output variable. A\n        :math:`(s_1, s_2, ..., s_n)`-shaped float array.\n\n    .. admonition:: Example\n\n        >>> x = np.random.uniform(-100, 100, (10, 20)).astype(np.float32)\n        >>> z = 10.0\n        >>> np.any(x < 0)\n        True\n        >>> np.any(x > z)\n        True\n        >>> y = F.clipped_relu(x, z=z)\n        >>> np.any(y.array < 0)\n        False\n        >>> np.any(y.array > z)\n        False\n\n    '
    (y,) = ClippedReLU(z).apply((x,))
    return y

def relu6(x):
    if False:
        print('Hello World!')
    'Rectifier Unit function clipped at 6.\n\n    It computes\n\n    .. math:: \\text{ReLU6}(x) = \\min(\\max(0, x), 6).\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable. A :math:`(s_1, s_2, ..., s_n)`-shaped float array.\n\n    Returns:\n        ~chainer.Variable: Output variable. A\n        :math:`(s_1, s_2, ..., s_n)`-shaped float array.\n\n    .. seealso:: :func:`chainer.functions.clipped_relu`\n\n    .. admonition:: Example\n\n        >>> x = np.array([-20, -2, 0, 2, 4, 10, 100]).astype(np.float32)\n        >>> x\n        array([-20.,  -2.,   0.,   2.,   4.,  10., 100.], dtype=float32)\n        >>> F.relu6(x)\n        variable([0., 0., 0., 2., 4., 6., 6.])\n\n    '
    (y,) = ClippedReLU(6.0).apply((x,))
    return y