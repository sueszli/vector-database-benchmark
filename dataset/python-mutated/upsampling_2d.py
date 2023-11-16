import numpy
from chainer.backends import cuda
from chainer import function_node
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
from chainer.utils import type_check

class Upsampling2D(pooling_2d.Pooling2D):
    """Upsampling over a set of 2d planes w/ indices used for max pooling."""

    def __init__(self, indexes, ksize, stride=None, pad=0, outsize=None, cover_all=True):
        if False:
            return 10
        super(Upsampling2D, self).__init__(ksize, stride, pad, cover_all)
        self.indexes = indexes
        (self.outh, self.outw) = (None, None) if outsize is None else outsize

    def check_type_forward(self, in_types):
        if False:
            print('Hello World!')
        n_in = in_types.size()
        type_check.expect(n_in == 1)
        x_type = in_types[0]
        type_check.expect(x_type.dtype.kind == 'f', x_type.ndim == 4, x_type.shape == self.indexes.shape)
        if self.outh is not None:
            expected_h = conv.get_conv_outsize(self.outh, self.kh, self.sy, self.ph, cover_all=self.cover_all)
            type_check.expect(x_type.shape[2] == expected_h)
        if self.outw is not None:
            expected_w = conv.get_conv_outsize(self.outw, self.kw, self.sx, self.pw, cover_all=self.cover_all)
            type_check.expect(x_type.shape[3] == expected_w)

    def forward_cpu(self, x):
        if False:
            return 10
        self._in_dtype = x[0].dtype
        (n, c, h, w) = x[0].shape
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(w, self.kw, self.sx, self.pw, cover_all=self.cover_all)
        up_y = numpy.zeros((n, c, self.outh, self.outw), dtype=self._in_dtype)
        up_y = conv.im2col_cpu(up_y, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all).transpose(0, 1, 4, 5, 2, 3)
        (colh, colw) = up_y.shape[2:4]
        up_y = up_y.reshape(-1, self.kh * self.kw)
        indexes = self.indexes.ravel()
        up_y[numpy.arange(len(indexes)), indexes] = x[0].ravel()
        up_y = up_y.reshape(n, c, colh, colw, self.kh, self.kw)
        up_y = conv.col2im_cpu(up_y.transpose(0, 1, 4, 5, 2, 3), self.sy, self.sx, self.ph, self.pw, self.outh, self.outw)
        return (up_y,)

    def forward_gpu(self, x):
        if False:
            while True:
                i = 10
        self._in_dtype = x[0].dtype
        xp = cuda.cupy
        (n, c, h, w) = x[0].shape
        if self.outh is None:
            self.outh = conv.get_deconv_outsize(h, self.kh, self.sy, self.ph, cover_all=self.cover_all)
        if self.outw is None:
            self.outw = conv.get_deconv_outsize(w, self.kw, self.sx, self.pw, cover_all=self.cover_all)
        up_y = xp.zeros((n, c, self.outh, self.outw), dtype=self._in_dtype)
        up_y = conv.im2col_gpu(up_y, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all)
        up_y = up_y.transpose(0, 1, 4, 5, 2, 3)
        (n, c, oy, ox, ky, kx) = up_y.shape
        indexes = xp.asarray(self.indexes, dtype=numpy.int32)
        cuda.elementwise('int32 index, T x, int32 n, int32 c, int32 oy, int32 ox,int32 ky, int32 kx', 'raw T up_y', '\n            int yn = i / c / oy / ox;\n            int yc = (i / oy / ox) % c;\n            int yoy = (i / ox) % oy;\n            int yox = i % ox;\n            up_y[yn * c * oy * ox * ky * kx +\n              yc * oy * ox * ky * kx +\n              yoy * ox * ky * kx +\n              yox * ky * kx +\n              index] = x;\n            ', 'upsampling_2d_fwd')(indexes, x[0], n, c, oy, ox, ky, kx, up_y)
        up_y = up_y.transpose(0, 1, 4, 5, 2, 3)
        up_y = conv.col2im_gpu(up_y, self.sy, self.sx, self.ph, self.pw, self.outh, self.outw)
        return (up_y,)

    def backward(self, indexes, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        return Upsampling2DGrad(self).apply(grad_outputs)

class Upsampling2DGrad(function_node.FunctionNode):

    def __init__(self, upsampling2d):
        if False:
            print('Hello World!')
        self.kh = upsampling2d.kh
        self.kw = upsampling2d.kw
        self.sy = upsampling2d.sy
        self.sx = upsampling2d.sx
        self.ph = upsampling2d.ph
        self.pw = upsampling2d.pw
        self.outh = upsampling2d.outh
        self.outw = upsampling2d.outw
        self.cover_all = upsampling2d.cover_all
        self.indexes = upsampling2d.indexes
        self._in_dtype = upsampling2d._in_dtype

    def forward_cpu(self, gy):
        if False:
            print('Hello World!')
        gcol = conv.im2col_cpu(gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all)
        (n, c, kh, kw, out_h, out_w) = gcol.shape
        gcol = gcol.transpose(0, 1, 4, 5, 2, 3).reshape(-1, kh * kw)
        indexes = self.indexes.ravel()
        gx = gcol[numpy.arange(len(indexes)), indexes]
        return (gx.reshape(n, c, out_h, out_w),)

    def forward_gpu(self, gy):
        if False:
            for i in range(10):
                print('nop')
        xp = cuda.cupy
        gcol = conv.im2col_gpu(gy[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all)
        gcol = gcol.transpose(0, 1, 4, 5, 2, 3)
        (n, c, oy, ox, ky, kx) = gcol.shape
        gcol = gcol.reshape((n, c, oy, ox, ky * kx))
        indexes = xp.asarray(self.indexes, dtype=numpy.int32)
        gx = xp.empty((n, c, oy, ox), dtype=self._in_dtype)
        cuda.elementwise('int32 indexes, raw T gcol, int32 n, int32 c, int32 oy,int32 ox, int32 ky, int32 kx', 'raw T gx', '\n            int ind_n = i / c / oy / ox;\n            int ind_c = (i / oy / ox) % c;\n            int ind_oy = (i / ox) % oy;\n            int ind_ox = i % ox;\n            int gcol_ky = indexes / kx;\n            int gcol_kx = indexes % kx;\n            float top_gx = gcol[ind_n * c * oy * ox * ky * kx +\n                                ind_c * oy * ox * ky * kx +\n                                ind_oy * ox * ky * kx +\n                                ind_ox * ky * kx +\n                                gcol_ky * kx +\n                                gcol_kx];\n            gx[ind_n * c * oy * ox +\n               ind_c * oy * ox +\n               ind_oy * ox +\n               ind_ox] = top_gx;\n            ', 'upsampling_2d_bwd')(indexes, gcol, n, c, oy, ox, ky, kx, gx)
        return (gx,)

    def backward(self, indexes, ggx):
        if False:
            while True:
                i = 10
        return Upsampling2D(self.indexes, (self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw), (self.outh, self.outw), self.cover_all).apply(ggx)

def upsampling_2d(x, indexes, ksize, stride=None, pad=0, outsize=None, cover_all=True):
    if False:
        while True:
            i = 10
    'Upsampling using pooling indices.\n\n    This function produces an upsampled image using pooling indices.\n\n    .. admonition:: Example\n\n        >>> x = np.arange(1, 37).reshape(1, 1, 6, 6).astype(np.float32)\n        >>> x = chainer.Variable(x)\n        >>> x.array\n        array([[[[ 1.,  2.,  3.,  4.,  5.,  6.],\n                 [ 7.,  8.,  9., 10., 11., 12.],\n                 [13., 14., 15., 16., 17., 18.],\n                 [19., 20., 21., 22., 23., 24.],\n                 [25., 26., 27., 28., 29., 30.],\n                 [31., 32., 33., 34., 35., 36.]]]], dtype=float32)\n\n        This is the original ``x`` before max pooling.\n\n        >>> pooled_x, indexes = F.max_pooling_2d(\n        ...     x, ksize=2, stride=2, return_indices=True)\n        >>> pooled_x.array\n        array([[[[ 8., 10., 12.],\n                 [20., 22., 24.],\n                 [32., 34., 36.]]]], dtype=float32)\n        >>> indexes\n        array([[[[3, 3, 3],\n                 [3, 3, 3],\n                 [3, 3, 3]]]])\n\n        These are the outputs from the max pooling operation including the\n        resulting indices that will be used to upsample ``pooled_x``. Note\n        that the indices all point to the largest, in the case the last,\n        elements in each window.\n\n        >>> upsampled_x = F.upsampling_2d(\n        ...     pooled_x, indexes, ksize=2, stride=2, outsize=x.shape[2:])\n        >>> upsampled_x.shape\n        (1, 1, 6, 6)\n        >>> upsampled_x.array\n        array([[[[ 0.,  0.,  0.,  0.,  0.,  0.],\n                 [ 0.,  8.,  0., 10.,  0., 12.],\n                 [ 0.,  0.,  0.,  0.,  0.,  0.],\n                 [ 0., 20.,  0., 22.,  0., 24.],\n                 [ 0.,  0.,  0.,  0.,  0.,  0.],\n                 [ 0., 32.,  0., 34.,  0., 36.]]]], dtype=float32)\n\n    Args:\n        x (~chainer.Variable): Input variable.\n        indexes (:ref:`ndarray`): Index array returned from\n            preceding call to :meth:`~chainer.functions.max_pooling_2d`.\n        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and\n            ``ksize=(k, k)`` are equivalent.\n        stride (int or pair of ints or None): Stride of pooling applications.\n            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is\n            specified, then it uses same stride as the pooling window size.\n        pad (int or pair of ints): Spatial padding width for the input array.\n            ``pad=p`` and ``pad=(p, p)`` are equivalent.\n        outsize ((int, int)): Expected output size (height, width).\n        cover_all (bool): Should be set to ``True`` if all spatial locations\n            were pooled into some output pixels during the preceding pooling\n            operation.  ``False`` otherwise. See\n            :meth:`~chainer.functions.max_pooling_2d`.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n    '
    return Upsampling2D(indexes, ksize, stride, pad, outsize, cover_all).apply((x,))[0]