import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer import configuration
from chainer import function_node
from chainer.utils import type_check
if cuda.cudnn_enabled:
    cudnn = cuda.cudnn
    libcudnn = cuda.cuda.cudnn

class GroupNormalization(function_node.FunctionNode):

    def __init__(self, groups, eps=1e-05):
        if False:
            i = 10
            return i + 15
        if not isinstance(groups, int):
            raise TypeError("Argument: 'groups' type must be (int).")
        self.groups = groups
        self.eps = eps
        self.mean = None
        self.inv_std = None
        self.dummy_gamma = None

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check.expect(in_types.size() == 3)
        (x_type, gamma_type, beta_type) = in_types
        type_check.expect(x_type.dtype.kind == 'f', x_type.ndim >= 2, gamma_type.ndim == 1, beta_type.ndim == 1, gamma_type.dtype.kind == 'f', gamma_type.dtype == beta_type.dtype, x_type.shape[1] == gamma_type.shape[0], gamma_type.shape == beta_type.shape)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        if inputs[0].shape[1] % self.groups != 0:
            raise ValueError("The number of channels {} is not divisible by 'groups' argument {}.".format(inputs[0].shape[1], self.groups))
        xp = backend.get_array_module(*inputs)
        if xp is cuda.cupy and chainer.should_use_cudnn('>=auto', 5000):
            return self.forward_cudnn(inputs)
        self.retain_inputs((0, 1))
        (x, gamma, beta) = inputs
        interm_dtype = numpy.promote_types(x.dtype, gamma.dtype)
        gamma = gamma.astype(interm_dtype, copy=False)
        beta = beta.astype(interm_dtype, copy=False)
        orig_shape = x.shape
        (batch_size, channels) = orig_shape[:2]
        groups = self.groups
        reduced_shape = (batch_size * groups, -1)
        x = x.reshape(reduced_shape)
        self.mean = x.mean(axis=1, dtype=interm_dtype)
        x_hat = x - self.mean[:, None]
        var = (x_hat * x_hat).mean(axis=1)
        var += self.eps
        self.inv_std = var
        del var
        xp.sqrt(self.inv_std, out=self.inv_std)
        xp.reciprocal(self.inv_std, out=self.inv_std)
        x_hat *= self.inv_std[:, None]
        y = x_hat.reshape((batch_size, channels, -1))
        y *= gamma[:, None]
        y += beta[:, None]
        y = y.reshape(orig_shape)
        return (y.astype(x.dtype, copy=False),)

    def forward_cudnn(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        if self.eps < libcudnn.CUDNN_BN_MIN_EPSILON:
            raise RuntimeError('cuDNN does not allow an eps value less than {}.'.format(libcudnn.CUDNN_BN_MIN_EPSILON))
        self.retain_inputs((0, 1))
        (x, gamma, beta) = inputs
        xp = cuda.cupy
        interm_dtype = numpy.promote_types(x.dtype, gamma.dtype)
        gamma = gamma.astype(interm_dtype, copy=False)
        beta = beta.astype(interm_dtype, copy=False)
        orig_shape = x.shape
        (batch_size, channels) = orig_shape[:2]
        groups = self.groups
        cudnn_shape = (1, batch_size * groups, -1, 1)
        x = x.reshape(cudnn_shape)
        with x.device:
            dummy_beta = xp.zeros(batch_size * groups, dtype=beta.dtype)
            self.dummy_gamma = xp.ones_like(dummy_beta)
        (x_hat, self.mean, self.inv_std) = cudnn.batch_normalization_forward_training(x, self.dummy_gamma, dummy_beta, dummy_beta, dummy_beta, None, None, self.eps, 1.0, True, libcudnn.CUDNN_BATCHNORM_SPATIAL, configuration.config.debug)
        y = x_hat.reshape((batch_size, channels, -1))
        cuda.elementwise('T gamma, T beta', 'U y', 'y = y * gamma + beta', 'groupnorm_y')(gamma[:, None], beta[:, None], y)
        y = y.reshape(orig_shape)
        return (y,)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        (x, gamma) = self.get_retained_inputs()
        (gy,) = grad_outputs
        interm_dtype = numpy.promote_types(x.dtype, gamma.dtype)
        gamma = chainer.functions.cast(gamma, interm_dtype)
        orig_shape = x.shape
        batch_size = orig_shape[0]
        groups = self.groups
        reduced_shape = (batch_size * groups, -1)
        x = x.reshape(reduced_shape)
        x_ = chainer.functions.cast(x, interm_dtype)
        (x_hat,) = _XHat(self.eps, self.mean, self.inv_std, self.dummy_gamma).apply((x_,))
        (gx_hat, ggamma, gbeta) = _ScaleShiftGrad().apply((x_hat, gamma, chainer.functions.cast(gy, interm_dtype)))
        (gx,) = _XHatGrad(self.eps, self.mean, self.inv_std, self.dummy_gamma, x_hat.array).apply((x_, gx_hat))
        gx = gx.reshape(orig_shape)
        return (chainer.functions.cast(gx, x.dtype), ggamma, gbeta)

class _ScaleShiftGrad(function_node.FunctionNode):

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        self.retain_inputs((0, 1, 2))
        (x_hat, gamma, gy) = inputs
        (batch_size, channels) = gy.shape[:2]
        gy = gy.reshape((batch_size, channels, -1))
        reduced_shape = x_hat.shape
        x_hat = x_hat.reshape((batch_size, channels, -1))
        gx_hat = gy * gamma[:, None]
        gbeta = gy.sum(axis=(0, 2))
        if backend.get_array_module(x_hat) is cuda.cupy:
            ggamma = cuda.reduce('T gy, T x_hat', 'T ggamma', 'gy * x_hat', 'a + b', 'ggamma = a', '0', 'groupnorm_ggamma')(gy, x_hat, axis=(0, 2))
        else:
            ggamma = (gy * x_hat).sum(axis=(0, 2))
        gx_hat = gx_hat.reshape(reduced_shape)
        return (gx_hat, ggamma, gbeta)

    def backward(self, indexes, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        (x_hat, gamma, gy) = self.get_retained_inputs()
        (ggx_hat, gggamma, ggbeta) = grad_outputs
        orig_shape = gy.shape
        (batch_size, channels) = gy.shape[:2]
        gy = gy.reshape((batch_size, channels, -1))
        reduced_shape = x_hat.shape
        x_hat = x_hat.reshape((batch_size, channels, -1))
        ggx_hat = ggx_hat.reshape((batch_size, channels, -1))
        gx_hat2 = gggamma[:, None] * gy
        ggamma2 = chainer.functions.sum(ggx_hat * gy, axis=(0, 2))
        ggy = ggx_hat * gamma[:, None] + gggamma[:, None] * x_hat + ggbeta[:, None]
        gx_hat2 = gx_hat2.reshape(reduced_shape)
        ggy = ggy.reshape(orig_shape)
        return (gx_hat2, ggamma2, ggy)

class _XHat(function_node.FunctionNode):

    def __init__(self, eps, mean, inv_std, dummy_gamma):
        if False:
            i = 10
            return i + 15
        self.eps = eps
        self.mean = mean
        self.inv_std = inv_std
        self.dummy_gamma = dummy_gamma

    def forward_cpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        self.retain_inputs((0,))
        (x,) = inputs
        x_hat = x - self.mean[:, None]
        x_hat *= self.inv_std[:, None]
        self.retain_outputs((0,))
        return (x_hat,)

    def forward_gpu(self, inputs):
        if False:
            print('Hello World!')
        self.retain_inputs((0,))
        (x,) = inputs
        x_hat = cuda.elementwise('T x, U mean, U inv_std', 'T x_hat', 'x_hat = (x - mean) * inv_std', 'groupnorm_x_hat')(x, self.mean[:, None], self.inv_std[:, None])
        self.retain_outputs((0,))
        return (x_hat,)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        (x,) = self.get_retained_inputs()
        (x_hat,) = self.get_retained_outputs()
        (gx_hat,) = grad_outputs
        return _XHatGrad(self.eps, self.mean, self.inv_std, self.dummy_gamma, x_hat.array).apply((x, gx_hat))

class _XHatGrad(function_node.FunctionNode):

    def __init__(self, eps, mean, inv_std, dummy_gamma, x_hat):
        if False:
            return 10
        self.eps = eps
        self.mean = mean
        self.inv_std = inv_std
        self.dummy_gamma = dummy_gamma
        self.x_hat = x_hat

    def forward(self, inputs):
        if False:
            return 10
        xp = backend.get_array_module(*inputs)
        if xp is cuda.cupy and chainer.should_use_cudnn('>=auto', 5000) and (self.dummy_gamma is not None):
            return self.forward_cudnn(inputs)
        self.retain_inputs((0, 1))
        (_, gx_hat) = inputs
        x_hat = self.x_hat
        self.x_hat = None
        gx_hat_avg = gx_hat.mean(axis=1, keepdims=True)
        gx_hat_x_hat_avg = (gx_hat * x_hat).mean(axis=1, keepdims=True)
        gx_std = gx_hat - gx_hat_avg - x_hat * gx_hat_x_hat_avg
        gx = self.inv_std[:, None] * gx_std
        self.retain_outputs((0,))
        return (gx,)

    def forward_cudnn(self, inputs):
        if False:
            i = 10
            return i + 15
        if self.eps < libcudnn.CUDNN_BN_MIN_EPSILON:
            raise RuntimeError('cuDNN does not allow an eps value less than {}.'.format(libcudnn.CUDNN_BN_MIN_EPSILON))
        self.retain_inputs((0, 1))
        (x, gx_hat) = inputs
        self.x_hat = None
        reduced_shape = x.shape
        cudnn_shape = (1,) + reduced_shape + (1,)
        x = x.reshape(cudnn_shape)
        gx_hat = gx_hat.reshape(cudnn_shape)
        (gx, _, _) = cudnn.batch_normalization_backward(x, self.dummy_gamma, gx_hat, self.mean, self.inv_std, self.eps, True, libcudnn.CUDNN_BATCHNORM_SPATIAL, configuration.config.debug)
        gx = gx.reshape(reduced_shape)
        self.retain_outputs((0,))
        return (gx,)

    def backward(self, indexes, grad_outputs):
        if False:
            return 10
        F = chainer.functions
        (x, gx_hat) = self.get_retained_inputs()
        (gx,) = self.get_retained_outputs()
        (ggx,) = grad_outputs
        (x_hat,) = _XHat(self.eps, self.mean, self.inv_std, self.dummy_gamma).apply((x,))
        ret = []
        if 0 in indexes:
            gx2l_std = x_hat * F.mean(ggx * gx, axis=1, keepdims=True)
            (gx2l,) = _MulInvStd(self.eps, self.mean, self.inv_std, self.dummy_gamma).apply((x, gx2l_std))
            gx_hat2r_std = ggx * F.mean(gx_hat * x_hat, axis=1, keepdims=True) + gx_hat * F.mean(ggx * x_hat, axis=1, keepdims=True)
            (gx_hat2r,) = _MulInvStd(self.eps, self.mean, self.inv_std, self.dummy_gamma).apply((x, gx_hat2r_std))
            (gx2r,) = _XHatGrad(self.eps, self.mean, self.inv_std, self.dummy_gamma, x_hat.array).apply((x, gx_hat2r))
            gx2 = -(gx2l + gx2r)
            ret.append(gx2)
        if 1 in indexes:
            (ggx_hat,) = _XHatGrad(self.eps, self.mean, self.inv_std, self.dummy_gamma, x_hat.array).apply((x, ggx))
            ret.append(ggx_hat)
        return ret

class _MulInvStd(function_node.FunctionNode):

    def __init__(self, eps, mean, inv_std, dummy_gamma):
        if False:
            for i in range(10):
                print('nop')
        self.eps = eps
        self.mean = mean
        self.inv_std = inv_std
        self.dummy_gamma = dummy_gamma

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        self.retain_inputs((0,))
        (_, y) = inputs
        z = self.inv_std[:, None] * y
        self.retain_outputs((0,))
        return (z,)

    def backward(self, indexes, grad_outputs):
        if False:
            i = 10
            return i + 15
        (x,) = self.get_retained_inputs()
        (z,) = self.get_retained_outputs()
        (gz,) = grad_outputs
        (x_hat,) = _XHat(self.eps, self.mean, self.inv_std, self.dummy_gamma).apply((x,))
        gx_std = x_hat * chainer.functions.mean(gz * z, axis=1, keepdims=True)
        (gx,) = _MulInvStd(self.eps, self.mean, self.inv_std, self.dummy_gamma).apply((x, gx_std))
        (gy,) = _MulInvStd(self.eps, self.mean, self.inv_std, self.dummy_gamma).apply((x, gz))
        return (-gx, gy)

def group_normalization(x, groups, gamma, beta, eps=1e-05):
    if False:
        print('Hello World!')
    'Group normalization function.\n\n    This function implements a "group normalization"\n    which divides the channels into groups and computes within each group\n    the mean and variance, then normalize by these statistics,\n    scales and shifts them.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`): Batch tensors.\n            First dimension of this value must be the size of minibatch and\n            second dimension must be the number of channels.\n            Moreover, this value must have one or more following dimensions,\n            such as height and width.\n        groups (int):\n            The number of channel groups.\n            This value must be a divisor of the number of channels.\n        gamma (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Scaling parameter.\n        beta (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Shifting parameter.\n        eps (float): Epsilon value for numerical stability of normalization.\n\n    Returns:\n        ~chainer.Variable: The output variable which has the same shape\n        as :math:`x`.\n\n    See: `Group Normalization <https://arxiv.org/abs/1803.08494>`_\n\n    .. seealso::\n\n        :class:`~chainer.links.GroupNormalization` to manage the model\n        parameters ``gamma`` and ``beta``.\n\n    '
    return GroupNormalization(groups, eps).apply((x, gamma, beta))[0]