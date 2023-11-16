import numpy
from chainer import backend
from chainer.functions.array import broadcast
from chainer.functions.array import concat
from chainer.functions.array import pad as pad_module
from chainer.functions.array import spatial_transformer_sampler
from chainer.functions.math import matmul

def deformable_convolution_2d_sampler(x, offset, W, b=None, stride=1, pad=0):
    if False:
        return 10
    'Two-dimensional deformable convolution function using computed offset.\n\n    This is an implementation of two-dimensional deformable convolution from\n    `Deformable Convolutional Networks <https://arxiv.org/abs/1703.06211>`_.\n\n    It takes four variables: the input image ``x``, the offset image\n    ``offset``, the filter weight ``W``, and the bias vector ``b``.\n\n    Notation: here is the notation for the dimensionalities.\n\n    - :math:`n` is the batch size.\n    - :math:`c_I` and :math:`c_O` are the number of the input and output,\n      respectively.\n    - :math:`h` and :math:`w` are the height and width of the input image,\n      respectively.\n    - :math:`k_H` and :math:`k_W` are the height and width of the filters,\n      respectively.\n    - :math:`s_Y` and :math:`s_X` are the strides of the filter.\n    - :math:`p_H` and :math:`p_W` are the spatial padding sizes.\n\n    The output size :math:`(h_O, w_O)` is determined by the following\n    equations:\n\n    .. math::\n\n       h_O &= (h + 2p_H - k_H) / s_Y + 1,\\\\\n       w_O &= (w + 2p_W - k_W) / s_X + 1.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable of shape :math:`(n, c_I, h, w)`.\n        offset (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Offset variable of shape\n            :math:`(n, 2 \\cdot k_H \\cdot k_W, h_O, w_O)`. The first\n            :math:`k_H \\cdot k_W` index of the second axis corresponds to\n            the offsets in the horizontal direction. The last\n            :math:`k_H \\cdot k_W` index of the second axis corresponds to\n            the offsets in the vertical direction.\n        W (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Weight variable of shape :math:`(c_O, c_I, k_H, k_W)`.\n        b (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Bias variable of length :math:`c_O` (optional).\n        stride (int or pair of ints): Stride of filter applications.\n            ``stride=s`` and ``stride=(s, s)`` are equivalent.\n        pad (int or pair of ints): Spatial padding width for input arrays.\n            ``pad=p`` and ``pad=(p, p)`` are equivalent.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    Deformable convolution adds 2D offsets to the regular grid sampling\n    locations in the standard convolution. It enables free form deformation of\n    the sampling grid.\n\n    See `Jifeng Dai, Haozhi Qi, Yuwen Xiong, Yi Li, Guodong Zhang, Han Hu,\n    Yichen Wei. Deformable Convolutional Networks\n    <https://arxiv.org/abs/1703.06211>`_\n\n    If the bias vector is given, then it is added to all spatial locations of\n    the output of convolution.\n\n    .. seealso::\n\n        :class:`~chainer.links.DeformableConvolution2D` to manage the model\n        parameters ``W`` and ``b``.\n\n    .. admonition:: Example\n\n        >>> x = np.random.uniform(0, 1, (2, 3, 4, 7)).astype(np.float32)\n        >>> offset = np.random.uniform(\n        ...     0, 1, (2, 2 * 3 * 3, 2, 5)).astype(np.float32)\n        >>> W = np.random.uniform(0, 1, (4, 3, 3, 3)).astype(np.float32)\n        >>> b = np.random.uniform(0, 1, (4,)).astype(np.float32)\n        >>> y = F.deformable_convolution_2d_sampler(x, offset, W, b)\n        >>> y.shape\n        (2, 4, 2, 5)\n\n    '
    (sy, sx) = _pair(stride)
    (ph, pw) = _pair(pad)
    (out_c, _, kh, kw) = W.shape
    (n, c, h, w) = x.shape
    (_, khkw2, out_h, out_w) = offset.shape
    if khkw2 != 2 * kh * kw:
        raise ValueError('The shape of the offset does not match the kernel size')
    grid = _offset2grid(offset, kh, kw, sy, sx, ph, pw, h, w)
    grid = grid.reshape(n, 2, kh * kw, out_h * out_w)
    x_pad = pad_module.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), 'constant')
    x_st = spatial_transformer_sampler.spatial_transformer_sampler(x_pad, grid)
    x_st = x_st.transpose(0, 3, 1, 2).reshape(n * out_h * out_w, c * kh * kw)
    W = W.transpose(1, 2, 3, 0).reshape(c * kh * kw, out_c)
    y = matmul.matmul(x_st, W)
    y = y.reshape(n, out_h, out_w, out_c).transpose(0, 3, 1, 2)
    if b is not None:
        b = broadcast.broadcast_to(b[None, :, None, None], y.shape)
        y += b
    return y

def _offset2grid(offset, kh, kw, sy, sx, ph, pw, h, w):
    if False:
        for i in range(10):
            print('nop')
    (n, khkw2, out_h, out_w) = offset.shape
    khkw = int(khkw2 / 2)
    xp = backend.get_array_module(offset)
    (ys, xs) = xp.meshgrid(xp.arange(0, sy * out_h, sy, dtype=numpy.float32), xp.arange(0, sx * out_w, sx, dtype=numpy.float32), indexing='ij', copy=False)
    filter_offset_x = xp.tile(xp.arange(kw, dtype=numpy.float32), kh)
    filter_offset_y = xp.repeat(xp.arange(kh, dtype=numpy.float32), kw)
    x_coord = offset[:, :khkw] + xs[None, None] + filter_offset_x[None, :, None, None]
    y_coord = offset[:, khkw:] + ys[None, None] + filter_offset_y[None, :, None, None]
    x_coord = (x_coord / (w + 2 * pw - 1) - 0.5) * 2
    y_coord = (y_coord / (h + 2 * ph - 1) - 0.5) * 2
    coord = concat.concat([x_coord, y_coord], axis=1)
    return coord

def _pair(x):
    if False:
        for i in range(10):
            print('nop')
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)