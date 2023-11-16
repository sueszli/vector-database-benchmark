import chainer
from chainer import backend
from chainer import function_node
from chainer.functions.activation import sigmoid
from chainer import utils
from chainer.utils import type_check

class SigmoidCrossEntropy(function_node.FunctionNode):
    """Sigmoid activation followed by a sigmoid cross entropy loss."""
    ignore_label = -1

    def __init__(self, normalize=True, reduce='mean'):
        if False:
            while True:
                i = 10
        self.normalize = normalize
        if reduce not in ('mean', 'no'):
            raise ValueError("only 'mean' and 'no' are valid for 'reduce', but '%s' is given" % reduce)
        self.reduce = reduce
        self.count = None

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        type_check._argname(in_types, ('x', 't'))
        (x_type, t_type) = in_types
        type_check.expect(x_type.dtype.kind == 'f', t_type.dtype.kind == 'i', x_type.shape == t_type.shape)

    def forward(self, inputs):
        if False:
            print('Hello World!')
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(*inputs)
        (x, t) = inputs
        self.ignore_mask = t != self.ignore_label
        loss = -(self.ignore_mask * (x * (t - (x >= 0)) - xp.log1p(xp.exp(-xp.abs(x)))))
        if not self.reduce == 'mean':
            return (utils.force_array(loss.astype(x.dtype)),)
        if self.normalize:
            count = xp.maximum(1, self.ignore_mask.sum())
        else:
            count = max(1, len(x))
        self.count = count
        return (utils.force_array(xp.divide(xp.sum(loss), self.count), dtype=x.dtype),)

    def backward(self, inputs, grad_outputs):
        if False:
            i = 10
            return i + 15
        (x, t) = self.get_retained_inputs()
        (gy,) = grad_outputs
        (gx,) = SigmoidCrossEntropyGrad(self.reduce, self.count, self.ignore_mask, t.data).apply((x, gy))
        return (gx, None)

class SigmoidCrossEntropyGrad(function_node.FunctionNode):
    """Sigmoid cross entropy gradient function."""

    def __init__(self, reduce, count, ignore_mask, t):
        if False:
            return 10
        self.reduce = reduce
        self.count = count
        self.ignore_mask = ignore_mask
        self.t = t

    def forward(self, inputs):
        if False:
            print('Hello World!')
        self.retain_inputs((0, 1))
        xp = backend.get_array_module(*inputs)
        (x, gy) = inputs
        (y,) = sigmoid.Sigmoid().forward((x,))
        if self.reduce == 'mean':
            gx = xp.divide(gy * self.ignore_mask * (y - self.t), self.count).astype(y.dtype)
        else:
            gx = (gy * self.ignore_mask * (y - self.t)).astype(y.dtype)
        return (gx,)

    def backward(self, indexes, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        (ggx,) = grad_outputs
        (x, gy) = self.get_retained_inputs()
        y = chainer.functions.sigmoid(x)
        yp = y * (1 - y)
        gx = yp * chainer.functions.broadcast_to(gy, yp.shape)
        ggy = y - self.t.astype(y.dtype)
        gx *= self.ignore_mask * ggx
        ggy *= self.ignore_mask * ggx
        if self.reduce == 'mean':
            gx /= self.count
            ggy = chainer.functions.sum(ggy) / self.count
        return (gx, ggy)

def sigmoid_cross_entropy(x, t, normalize=True, reduce='mean'):
    if False:
        i = 10
        return i + 15
    "Computes cross entropy loss for pre-sigmoid activations.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            A variable object holding a matrix whose\n            (i, j)-th element indicates the unnormalized log probability of\n            the j-th unit at the i-th example.\n        t (:class:`~chainer.Variable` or :ref:`ndarray`):\n            A variable object holding a matrix whose\n            (i, j)-th element indicates a signed integer vector of\n            ground truth labels 0 or 1.\n            If ``t[i, j] == -1``, corresponding ``x[i, j]`` is ignored.\n            Loss is zero if all ground truth labels are ``-1``.\n        normalize (bool): Variable holding a boolean value which\n            determines the normalization constant. If true, this function\n            normalizes the cross entropy loss across all instances. If else,\n            it only normalizes along a batch size.\n        reduce (str): Variable holding a ``str`` which\n            determines whether to reduce the shape of the input.\n            If it is ``'mean'``, it computes the sum of cross entropy\n            and normalize it according to ``normalize`` option.\n            If is is ``'no'``, this function computes cross entropy for each\n            instance and does not normalize it (``normalize`` option is\n            ignored). In this case, the loss value of the ignored instance,\n            which has ``-1`` as its target value, is set to ``0``.\n\n    Returns:\n        ~chainer.Variable: A variable object holding an array of the cross\n        entropy.\n        If ``reduce`` is ``'mean'``, it is a scalar array.\n        If ``reduce`` is ``'no'``, the shape is same as those of ``x`` and\n        ``t``.\n\n    .. note::\n\n       This function is differentiable only by ``x``.\n\n    .. admonition:: Example\n\n        >>> x = np.array([[-2.0, 3.0, 0.5], [5.0, 2.0, -0.5]]).astype(np.float32)\n        >>> x\n        array([[-2. ,  3. ,  0.5],\n               [ 5. ,  2. , -0.5]], dtype=float32)\n        >>> t = np.array([[0, 1, 0], [1, 1, -1]]).astype(np.int32)\n        >>> t\n        array([[ 0,  1,  0],\n               [ 1,  1, -1]], dtype=int32)\n        >>> F.sigmoid_cross_entropy(x, t)\n        variable(0.25664714)\n        >>> F.sigmoid_cross_entropy(x, t, normalize=False)\n        variable(0.64161783)\n        >>> y = F.sigmoid_cross_entropy(x, t, reduce='no')\n        >>> y.shape\n        (2, 3)\n        >>> y.array\n        array([[ 0.126928  ,  0.04858735,  0.974077  ],\n               [ 0.00671535,  0.126928  , -0.        ]], dtype=float32)\n\n    "
    return SigmoidCrossEntropy(normalize, reduce).apply((x, t))[0]