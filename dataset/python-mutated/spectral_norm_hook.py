import paddle
from .. import functional as F
from ..layer.common import Linear
from ..layer.conv import Conv1DTranspose, Conv2DTranspose, Conv3DTranspose
__all__ = []

def normal_(x, mean=0.0, std=1.0):
    if False:
        return 10
    temp_value = paddle.normal(mean, std, shape=x.shape)
    paddle.assign(temp_value, x)
    return x

class SpectralNorm:

    def __init__(self, name='weight', n_power_iterations=1, dim=0, eps=1e-12):
        if False:
            print('Hello World!')
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(f'Expected n_power_iterations to be positive, but got n_power_iterations={n_power_iterations}')
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight):
        if False:
            i = 10
            return i + 15
        weight_mat = weight
        if self.dim != 0:
            weight_mat = weight_mat.transpose([self.dim] + [d for d in range(weight_mat.dim()) if d != self.dim])
        height = weight_mat.shape[0]
        return weight_mat.reshape([height, -1])

    def compute_weight(self, layer, do_power_iteration):
        if False:
            i = 10
            return i + 15
        weight = getattr(layer, self.name + '_orig')
        u = getattr(layer, self.name + '_u')
        v = getattr(layer, self.name + '_v')
        weight_mat = self.reshape_weight_to_matrix(weight)
        if do_power_iteration:
            with paddle.no_grad():
                for _ in range(self.n_power_iterations):
                    paddle.assign(F.normalize(paddle.matmul(weight_mat, u, transpose_x=True, transpose_y=False), axis=0, epsilon=self.eps), v)
                    paddle.assign(F.normalize(paddle.matmul(weight_mat, v), axis=0, epsilon=self.eps), u)
                if self.n_power_iterations > 0:
                    u = u.clone()
                    v = v.clone()
        sigma = paddle.dot(u, paddle.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def __call__(self, layer, inputs):
        if False:
            i = 10
            return i + 15
        setattr(layer, self.name, self.compute_weight(layer, do_power_iteration=layer.training))

    @staticmethod
    def apply(layer, name, n_power_iterations, dim, eps):
        if False:
            while True:
                i = 10
        for (k, hook) in layer._forward_pre_hooks.items():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(f'Cannot register two spectral_norm hooks on the same parameter {name}')
        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = layer._parameters[name]
        with paddle.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)
            (h, w) = weight_mat.shape
            u = layer.create_parameter([h])
            u = normal_(u, 0.0, 1.0)
            v = layer.create_parameter([w])
            v = normal_(v, 0.0, 1.0)
            u = F.normalize(u, axis=0, epsilon=fn.eps)
            v = F.normalize(v, axis=0, epsilon=fn.eps)
        del layer._parameters[fn.name]
        layer.add_parameter(fn.name + '_orig', weight)
        setattr(layer, fn.name, weight * 1.0)
        layer.register_buffer(fn.name + '_u', u)
        layer.register_buffer(fn.name + '_v', v)
        layer.register_forward_pre_hook(fn)
        return fn

def spectral_norm(layer, name='weight', n_power_iterations=1, eps=1e-12, dim=None):
    if False:
        print('Hello World!')
    "\n    Applies spectral normalization to a parameter according to the\n    following Calculation:\n\n    Step 1:\n    Generate vector U in shape of [H], and V in shape of [W].\n    While H is the :attr:`dim` th dimension of the input weights,\n    and W is the product result of remaining dimensions.\n\n    Step 2:\n    :attr:`n_power_iterations` should be a positive integer, do following\n    calculations with U and V for :attr:`power_iters` rounds.\n\n    .. math::\n\n        \\mathbf{v} := \\frac{\\mathbf{W}^{T} \\mathbf{u}}{\\|\\mathbf{W}^{T} \\mathbf{u}\\|_2}\n\n        \\mathbf{u} := \\frac{\\mathbf{W} \\mathbf{v}}{\\|\\mathbf{W} \\mathbf{v}\\|_2}\n\n    Step 3:\n    Calculate :math:`\\sigma(\\mathbf{W})` and normalize weight values.\n\n    .. math::\n\n        \\sigma(\\mathbf{W}) = \\mathbf{u}^{T} \\mathbf{W} \\mathbf{v}\n\n        \\mathbf{W} = \\frac{\\mathbf{W}}{\\sigma(\\mathbf{W})}\n\n\n    Refer to `Spectral Normalization <https://arxiv.org/abs/1802.05957>`_ .\n\n    Parameters:\n        layer(Layer): Layer of paddle, which has weight.\n        name(str, optional): Name of the weight parameter. Default: 'weight'.\n        n_power_iterations(int, optional): The number of power iterations to calculate spectral norm. Default: 1.\n        eps(float, optional): The epsilon for numerical stability in calculating norms. Default: 1e-12.\n        dim(int, optional): The index of dimension which should be permuted to the first before reshaping Input(Weight) to matrix, it should be set as 0 if Input(Weight) is the weight of fc layer, and should be set as 1 if Input(Weight) is the weight of conv layer. Default: None.\n\n    Returns:\n        Layer, the original layer with the spectral norm hook.\n\n    Examples:\n        .. code-block:: python\n\n            >>> from paddle.nn import Conv2D\n            >>> from paddle.nn.utils import spectral_norm\n            >>> paddle.seed(2023)\n            >>> conv = Conv2D(3, 1, 3)\n            >>> sn_conv = spectral_norm(conv)\n            >>> print(sn_conv)\n            Conv2D(3, 1, kernel_size=[3, 3], data_format=NCHW)\n            >>> # Conv2D(3, 1, kernel_size=[3, 3], data_format=NCHW)\n            >>> print(sn_conv.weight)\n            Tensor(shape=[1, 3, 3, 3], dtype=float32, place=Place(cpu), stop_gradient=False,\n            [[[[ 0.01668976,  0.30305523,  0.11405435],\n               [-0.06765547, -0.50396705, -0.40925547],\n               [ 0.47344422,  0.03628403,  0.45277366]],\n              [[-0.15177251, -0.16305730, -0.15723954],\n               [-0.28081197, -0.09183260, -0.08081978],\n               [-0.40895155,  0.18298769, -0.29325116]],\n              [[ 0.21819633, -0.01822380, -0.50351536],\n               [-0.06262003,  0.17713565,  0.20517939],\n               [ 0.16659889, -0.14333329,  0.05228264]]]])\n\n    "
    if dim is None:
        if isinstance(layer, (Conv1DTranspose, Conv2DTranspose, Conv3DTranspose, Linear)):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(layer, name, n_power_iterations, dim, eps)
    return layer