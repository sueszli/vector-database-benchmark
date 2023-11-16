from six import moves
import chainer
from chainer import backend
from chainer import function_node
from chainer.utils import type_check
from chainer import variable

def _pair(x):
    if False:
        return 10
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)

class LocalConvolution2DFunction(function_node.FunctionNode):

    def __init__(self, stride=1):
        if False:
            return 10
        (self.sy, self.sx) = _pair(stride)

    def check_type_forward(self, in_types):
        if False:
            while True:
                i = 10
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        (x_type, w_type) = in_types[:2]
        type_check.expect(x_type.dtype.kind == 'f', w_type.dtype.kind == 'f', x_type.ndim == 4, w_type.ndim == 6, x_type.shape[1] == w_type.shape[3])
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(b_type.dtype == x_type.dtype, b_type.ndim == 3, b_type.shape == w_type.shape[:3])

    def forward(self, inputs):
        if False:
            while True:
                i = 10
        (x, W) = inputs[:2]
        b = inputs[2] if len(inputs) == 3 else None
        (stride_row, stride_col) = (self.sy, self.sx)
        (output_row, output_col) = (W.shape[1], W.shape[2])
        feature_dim = W.shape[3] * W.shape[4] * W.shape[5]
        xp = backend.get_array_module(*inputs)
        output = xp.empty((x.shape[0], W.shape[0], output_row, output_col), dtype=x.dtype)
        for i in moves.range(output_row):
            for j in moves.range(output_col):
                slice_row = slice(i * stride_row, i * stride_row + W.shape[4])
                slice_col = slice(j * stride_col, j * stride_col + W.shape[5])
                x_flatten = xp.reshape(x[..., slice_row, slice_col], (-1, feature_dim))
                W_flatten = xp.reshape(W[:, i, j, ...], (-1, feature_dim))
                output[..., i, j] = xp.dot(x_flatten, W_flatten.T)
        if b is not None:
            output += b[None, :, :, :]
        self.retain_inputs((0, 1))
        return (output,)

    def backward(self, indices, grad_outputs):
        if False:
            return 10
        (xvar, Wvar) = self.get_retained_inputs()
        x = xvar.data
        W = Wvar.data
        (gyvar,) = grad_outputs
        gy = gyvar.data
        xp = backend.get_array_module(x, W)
        (stride_row, stride_col) = (self.sy, self.sx)
        (output_row, output_col) = (W.shape[1], W.shape[2])
        ret = []
        if 0 in indices:
            gx = xp.zeros_like(x)
            for i in moves.range(output_row):
                for j in moves.range(output_col):
                    slice_row = slice(i * stride_row, i * stride_row + W.shape[4])
                    slice_col = slice(j * stride_col, j * stride_col + W.shape[5])
                    W_slice = W[:, i, j, ...]
                    gy_slice = gy[..., i, j]
                    gx[:, :, slice_row, slice_col] += xp.tensordot(gy_slice, W_slice, axes=[(1,), (0,)])
            ret.append(chainer.functions.cast(variable.as_variable(gx), x.dtype))
        if 1 in indices:
            gW = xp.empty_like(W)
            for i in moves.range(output_row):
                for j in moves.range(output_col):
                    slice_row = slice(i * stride_row, i * stride_row + W.shape[4])
                    slice_col = slice(j * stride_col, j * stride_col + W.shape[5])
                    x_slice = x[:, :, slice_row, slice_col]
                    gy_slice = gy[:, :, i, j]
                    gW[:, i, j, :, :, :] = xp.tensordot(gy_slice, x_slice, axes=[(0,), (0,)])
            ret.append(chainer.functions.cast(variable.as_variable(gW), W.dtype))
        if 2 in indices:
            gb = chainer.functions.sum(gyvar, axis=0)
            ret.append(gb)
        return ret

def local_convolution_2d(x, W, b=None, stride=1):
    if False:
        for i in range(10):
            print('nop')
    'Two-dimensional local convolution function.\n\n    Locally-connected function for 2D inputs. Works similarly to\n    convolution_2d, except that weights are unshared, that is, a different set\n    of filters is applied at each different patch of the input.\n    It takes two or three variables: the input image ``x``, the filter weight\n    ``W``, and optionally, the bias vector ``b``.\n\n    Notation: here is a notation for dimensionalities.\n\n    - :math:`n` is the batch size.\n    - :math:`c_I` is the number of the input.\n    - :math:`c_O` is the number of output channels.\n    - :math:`h` and :math:`w` are the height and width of the input image,\n      respectively.\n    - :math:`h_O` and :math:`w_O` are the height and width of the output image,\n      respectively.\n    - :math:`k_H` and :math:`k_W` are the height and width of the filters,\n      respectively.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable of shape :math:`(n, c_I, h, w)`.\n        W (:class:`~chainer.Variable` or :ref:`ndarray`): Weight variable of\n            shape :math:`(c_O, h_O, w_O, c_I, k_H, k_W)`.\n        b (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Bias variable of shape :math:`(c_O,h_O,w_O)` (optional).\n        stride (int or pair of ints): Stride of filter applications.\n            ``stride=s`` and ``stride=(s, s)`` are equivalent.\n\n\n    Returns:\n        ~chainer.Variable:\n            Output variable. Its shape is :math:`(n, c_O, h_O, w_O)`.\n\n    Like ``Convolution2D``, ``LocalConvolution2D`` function computes\n    correlations between filters and patches of size :math:`(k_H, k_W)` in\n    ``x``.\n    But unlike ``Convolution2D``, ``LocalConvolution2D`` has a separate filter\n    for each patch of the input\n\n    :math:`(h_O, w_O)` is determined by the equivalent equation of\n    ``Convolution2D``, without any padding\n\n    If the bias vector is given, then it is added to all spatial locations of\n    the output of convolution.\n\n    .. seealso::\n\n        :class:`~chainer.links.LocalConvolution2D` to manage the model\n        parameters ``W`` and ``b``.\n\n    .. admonition:: Example\n\n        >>> x = np.random.uniform(0, 1, (2, 3, 7, 7))\n        >>> W = np.random.uniform(0, 1, (2, 5, 5, 3, 3, 3))\n        >>> b = np.random.uniform(0, 1, (2, 5, 5))\n        >>> y = F.local_convolution_2d(x, W, b)\n        >>> y.shape\n        (2, 2, 5, 5)\n\n    '
    fnode = LocalConvolution2DFunction(stride)
    if b is None:
        args = (x, W)
    else:
        args = (x, W, b)
    (y,) = fnode.apply(args)
    return y