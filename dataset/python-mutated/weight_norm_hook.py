import paddle
from paddle import _C_ops
from ...base.data_feeder import check_variable_and_dtype
from ...base.layer_helper import LayerHelper
from ...framework import in_dynamic_mode
__all__ = []

def l2_norm(x, axis, epsilon=1e-12, name=None):
    if False:
        i = 10
        return i + 15
    if len(x.shape) == 1:
        axis = 0
    if in_dynamic_mode():
        (out, norm) = _C_ops.norm(x, 1 if axis is None else axis, epsilon, False)
        return paddle.squeeze(norm, axis=[axis])
    check_variable_and_dtype(x, 'X', ('float32', 'float64'), 'norm')
    helper = LayerHelper('l2_normalize', **locals())
    out = helper.create_variable_for_type_inference(dtype=x.dtype)
    norm = helper.create_variable_for_type_inference(dtype=x.dtype)
    helper.append_op(type='norm', inputs={'X': x}, outputs={'Out': out, 'Norm': norm}, attrs={'axis': 1 if axis is None else axis, 'epsilon': epsilon})
    return paddle.squeeze(norm, axis=[axis])

def norm_except_dim(p, dim):
    if False:
        i = 10
        return i + 15
    shape = p.shape
    ndims = len(shape)
    if dim == -1:
        return paddle.sqrt(paddle.sum(paddle.square(p)) + 1e-12)
    elif dim == 0:
        p_matrix = paddle.reshape(p, (shape[0], -1))
        return l2_norm(p_matrix, axis=1)
    elif dim == ndims - 1:
        p_matrix = paddle.reshape(p, (-1, shape[-1]))
        return l2_norm(p_matrix, axis=0)
    else:
        perm = list(range(ndims))
        perm[0] = dim
        perm[dim] = 0
        p_transposed = paddle.transpose(p, perm)
        return norm_except_dim(p_transposed, 0)

def _weight_norm(v, g, dim):
    if False:
        i = 10
        return i + 15
    shape = v.shape
    ndims = len(shape)
    if dim == -1:
        v_normalized = v / (paddle.sqrt(paddle.sum(paddle.square(v))) + 1e-12)
    elif dim == 0:
        p_matrix = paddle.reshape(v, (shape[0], -1))
        v_normalized = paddle.nn.functional.normalize(p_matrix, axis=1)
        v_normalized = paddle.reshape(v_normalized, shape)
    elif dim == ndims - 1:
        p_matrix = paddle.reshape(v, (-1, shape[-1]))
        v_normalized = paddle.nn.functional.normalize(p_matrix, axis=0)
        v_normalized = paddle.reshape(v_normalized, shape)
    else:
        perm = list(range(ndims))
        perm[0] = dim
        perm[dim] = 0
        p_transposed = paddle.transpose(v, perm)
        transposed_shape = p_transposed.shape
        p_matrix = paddle.reshape(p_transposed, (p_transposed.shape[0], -1))
        v_normalized = paddle.nn.functional.normalize(p_matrix, axis=1)
        v_normalized = paddle.reshape(v_normalized, transposed_shape)
        v_normalized = paddle.transpose(v_normalized, perm)
    weight = paddle.tensor.math._multiply_with_axis(v_normalized, g, axis=dim if dim is not None else -1)
    return weight

class WeightNorm:

    def __init__(self, name, dim):
        if False:
            return 10
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    def compute_weight(self, layer):
        if False:
            print('Hello World!')
        g = getattr(layer, self.name + '_g')
        v = getattr(layer, self.name + '_v')
        return _weight_norm(v, g, self.dim)

    @staticmethod
    def apply(layer, name, dim):
        if False:
            i = 10
            return i + 15
        for (k, hook) in layer._forward_pre_hooks.items():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(f'Cannot register two weight_norm hooks on the same parameter {name}')
        if dim is None:
            dim = -1
        weight_dim = len(layer._parameters[name].shape)
        assert dim < weight_dim and dim >= -1 * weight_dim, 'dim must set between [-R, R), R means the dimension of weight.'
        if dim != -1:
            dim = (dim + weight_dim) % weight_dim
        fn = WeightNorm(name, dim)
        w = getattr(layer, name)
        del layer._parameters[name]
        g_var = norm_except_dim(w, dim)
        v = layer.create_parameter(w.shape, dtype=w.dtype)
        layer.add_parameter(name + '_v', v)
        g = layer.create_parameter(g_var.shape, dtype=g_var.dtype)
        layer.add_parameter(name + '_g', g)
        with paddle.no_grad():
            paddle.assign(w, v)
            paddle.assign(g_var, g)
        setattr(layer, name, fn.compute_weight(layer))
        layer.register_forward_pre_hook(fn)
        return fn

    def remove(self, layer):
        if False:
            return 10
        w_var = self.compute_weight(layer)
        delattr(layer, self.name)
        del layer._parameters[self.name + '_g']
        del layer._parameters[self.name + '_v']
        w = layer.create_parameter(w_var.shape, dtype=w_var.dtype)
        layer.add_parameter(self.name, w)
        with paddle.no_grad():
            paddle.assign(w_var, w)

    def __call__(self, layer, inputs):
        if False:
            return 10
        setattr(layer, self.name, self.compute_weight(layer))

def weight_norm(layer, name='weight', dim=0):
    if False:
        i = 10
        return i + 15
    "\n    Applies weight normalization to a parameter according to the\n    following formula:\n\n    .. math::\n\n        \\mathbf{w} = g \\dfrac{v}{\\|v\\|}\n\n    Weight normalization is a reparameterization of the weight vectors in a neural network that\n    decouples the magnitude of those weight vectors from their direction. Weight normalization\n    replaces the parameter specified by ``name`` (eg: 'weight') with two parameters: one parameter\n    specifying the magnitude (eg: 'weight_g') and one parameter specifying the direction\n    (eg: 'weight_v'). Weight normalization has been implemented as discussed in this paper:\n\n    `Weight Normalization: A Simple Reparameterization to Accelerate Training of Deep Neural Networks\n    <https://arxiv.org/pdf/1602.07868.pdf>`_.\n\n    Parameters:\n        layer(Layer): Layer of paddle, which has weight.\n        name(str, optional): Name of the weight parameter. Default: 'weight'.\n        dim(int, optional): Dimension over which to compute the norm. Dim is a non-negative number\n              which is less than the rank of weight Tensor. For Example, dim can be chosen from 0,\n              1, 2, 3 for convolution whose weight shape is [cout, cin, kh, kw] and rank is 4.\n              If dim is set to None, meaning that all elements will be normalized. Default: 0.\n\n    Returns:\n        Origin layer with weight norm hook.\n\n    Examples:\n        .. code-block:: python\n\n          >>> from paddle.nn import Conv2D\n          >>> from paddle.nn.utils import weight_norm\n\n          >>> conv = Conv2D(3, 5, 3)\n          >>> wn = weight_norm(conv)\n          >>> print(conv.weight_g.shape)\n          [5]\n          >>> print(conv.weight_v.shape)\n          [5, 3, 3, 3]\n    "
    WeightNorm.apply(layer, name, dim)
    return layer

def remove_weight_norm(layer, name='weight'):
    if False:
        i = 10
        return i + 15
    "\n    remove weight normalization from layer.\n\n    Parameters:\n        layer(Layer): Layer of paddle, which has weight.\n        name(str, optional): Name of the weight parameter. Default: 'weight'.\n\n    Returns:\n        Layer, the origin layer without weight norm\n\n    Examples:\n        .. code-block:: python\n\n            >>> import paddle\n            >>> from paddle.nn import Conv2D\n            >>> from paddle.nn.utils import weight_norm, remove_weight_norm\n            >>> paddle.seed(2023)\n\n            >>> conv = Conv2D(3, 5, 3)\n            >>> wn = weight_norm(conv)\n            >>> print(conv.weight_g)\n            Parameter containing:\n            Tensor(shape=[5], dtype=float32, place=Place(cpu), stop_gradient=False,\n                   [1.35883713, 1.32126212, 1.56303072, 1.20874095, 1.22893476])\n            >>> remove_weight_norm(conv)\n            >>> # The following is the effect after removing the weight norm:\n            >>> # print(conv.weight_g)\n            >>> # AttributeError: 'Conv2D' object has no attribute 'weight_g'\n    "
    for (k, hook) in layer._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(layer)
            del layer._forward_pre_hooks[k]
            return layer
    raise ValueError(f"weight_norm of '{name}' not found in {layer}")