import numpy
from chainer import function_node
from chainer.utils.conv import col2im_cpu
from chainer.utils.conv import col2im_gpu
from chainer.utils.conv import im2col_cpu
from chainer.utils.conv import im2col_gpu
from chainer.utils import type_check

def _pair(x):
    if False:
        i = 10
        return i + 15
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)

def _col2im(x, *args, **kwargs):
    if False:
        while True:
            i = 10
    if isinstance(x, numpy.ndarray):
        return col2im_cpu(x, *args, **kwargs)
    return col2im_gpu(x, *args, **kwargs)

def _im2col(x, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, numpy.ndarray):
        return im2col_cpu(x, *args, **kwargs)
    return im2col_gpu(x, *args, **kwargs)

class Im2Col(function_node.FunctionNode):
    """Im2Col function."""

    def __init__(self, ksize, stride, pad, cover_all, dilate):
        if False:
            print('Hello World!')
        (self.kh, self.kw) = _pair(ksize)
        (self.sy, self.sx) = _pair(stride)
        (self.ph, self.pw) = _pair(pad)
        (self.dy, self.dx) = _pair(dilate)
        self.cover_all = cover_all

    def check_type_forward(self, in_types):
        if False:
            for i in range(10):
                print('nop')
        type_check._argname(in_types, ('x',))
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f', x_type.ndim == 4)

    def forward(self, inputs):
        if False:
            i = 10
            return i + 15
        (x,) = inputs
        y = _im2col(x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        (n, c, kh, kw, out_h, out_w) = y.shape
        return (y.reshape(n, c * kh * kw, out_h, out_w),)

    def backward(self, indexes, grad_outputs):
        if False:
            while True:
                i = 10
        return Im2ColGrad((self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw), self.cover_all, (self.dy, self.dx), self.inputs[0].shape).apply(grad_outputs)

class Im2ColGrad(function_node.FunctionNode):
    """Im2Col gradient function."""

    def __init__(self, ksize, stride, pad, cover_all, dilate, in_shape):
        if False:
            for i in range(10):
                print('nop')
        (self.kh, self.kw) = _pair(ksize)
        (self.sy, self.sx) = _pair(stride)
        (self.ph, self.pw) = _pair(pad)
        (self.dy, self.dx) = _pair(dilate)
        self.cover_all = cover_all
        self.in_shape = in_shape

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        type_check._argname(in_types, ('gy',))
        gy_type = in_types[0]
        type_check.expect(gy_type.dtype.kind == 'f', gy_type.ndim == 4)

    def forward(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        (_, c, h, w) = self.in_shape
        (gy,) = inputs
        (n, _, out_h, out_w) = gy.shape
        gy = gy.reshape(n, c, self.kh, self.kw, out_h, out_w)
        gx = _col2im(gy, self.sy, self.sx, self.ph, self.pw, h, w, self.dy, self.dx)
        return (gx,)

    def backward(self, indexes, grad_outputs):
        if False:
            i = 10
            return i + 15
        return Im2Col((self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw), self.cover_all, (self.dy, self.dx)).apply(grad_outputs)

def im2col(x, ksize, stride=1, pad=0, cover_all=False, dilate=1):
    if False:
        print('Hello World!')
    'Extract patches from an image based on the filter.\n\n    This function rearranges patches of an image and puts them in the channel\n    dimension of the output.\n\n    Patches are extracted at positions shifted by multiples of ``stride`` from\n    the first position ``-pad`` for each spatial axis.\n    The right-most (or bottom-most) patches do not run over the padded spatial\n    size.\n\n    Notation: here is a notation.\n\n    - :math:`n` is the batch size.\n    - :math:`c` is the number of the input channels.\n    - :math:`h` and :math:`w` are the height and width of the input image,\n      respectively.\n    - :math:`k_H` and :math:`k_W` are the height and width of the filters,\n      respectively.\n    - :math:`s_Y` and :math:`s_X` are the strides of the filter.\n    - :math:`p_H` and :math:`p_W` are the spatial padding sizes.\n    - :math:`d_Y` and :math:`d_X` are the dilation factors of filter         application.\n\n    The output size :math:`(h_O, w_O)` is determined by the following\n    equations when ``cover_all = False``:\n\n    .. math::\n\n       h_O &= (h + 2p_H - k_H - (k_H - 1) * (d_Y - 1)) / s_Y + 1,\\\\\n       w_O &= (w + 2p_W - k_W - (k_W - 1) * (d_X - 1)) / s_X + 1.\n\n    When ``cover_all = True``, the output size is determined by\n    the following equations:\n\n    .. math::\n\n       h_O &= (h + 2p_H - k_H - (k_H - 1) * (d_Y - 1) + s_Y - 1) / s_Y + 1,\\\\\n       w_O &= (w + 2p_W - k_W - (k_W - 1) * (d_X - 1) + s_X - 1) / s_X + 1.\n\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable of shape :math:`(n, c, h, w)`.\n        ksize (int or pair of ints): Size of filters (a.k.a. kernels).\n            ``ksize=k`` and ``ksize=(k, k)`` are equivalent.\n        stride (int or pair of ints): Stride of filter applications.\n            ``stride=s`` and ``stride=(s, s)`` are equivalent.\n        pad (int or pair of ints): Spatial padding width for input arrays.\n            ``pad=p`` and ``pad=(p, p)`` are equivalent.\n        cover_all (bool): If ``True``, all spatial locations are rearranged\n            into some output pixels. It may make the output size larger.\n        dilate (int or pair of ints): Dilation factor of filter applications.\n            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.\n\n    Returns:\n        ~chainer.Variable:\n        Output variable whose shape is\n        :math:`(n, c \\cdot k_H \\cdot k_W, h_O, w_O)`\n\n    '
    return Im2Col(ksize, stride, pad, cover_all, dilate).apply((x,))[0]