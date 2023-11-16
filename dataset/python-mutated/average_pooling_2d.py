import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import function_node
from chainer.functions.pooling import average_pooling_nd
from chainer.functions.pooling import pooling_2d
from chainer.utils import conv
import chainerx

class AveragePooling2D(pooling_2d.Pooling2D):
    """Average pooling over a set of 2d planes."""

    def forward_cpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        if intel64.should_use_ideep('>=auto') and intel64.inputs_all_ready(x):
            return self._forward_ideep(x)
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        col = conv.im2col_cpu(x[0], self.kh, self.kw, self.sy, self.sx, self.ph, self.pw)
        y = col.mean(axis=(2, 3))
        return (y,)

    def _forward_ideep(self, x):
        if False:
            return 10
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        self.retain_inputs((0,))
        (n, c, h, w) = x[0].shape
        y_h = conv.get_conv_outsize(h, self.kh, self.sy, self.ph, self.cover_all)
        assert y_h > 0, 'Height in the output should be positive.'
        y_w = conv.get_conv_outsize(w, self.kw, self.sx, self.pw, self.cover_all)
        assert y_w > 0, 'Width in the output should be positive.'
        pd = self.sy * (y_h - 1) + self.kh - h - self.ph
        pr = self.sx * (y_w - 1) + self.kw - w - self.pw
        pp = intel64.ideep.pooling2DParam((n, c, y_h, y_w), self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, pd, pr, intel64.ideep.pooling2DParam.pooling_avg_include_padding)
        (y,) = intel64.ideep.pooling2D.Forward(intel64.ideep.array(x[0]), pp)
        return (y,)

    def forward_gpu(self, x):
        if False:
            return 10
        if chainer.should_use_cudnn('>=auto'):
            self.retain_inputs((0,))
            return super(AveragePooling2D, self).forward_gpu(x)
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        (n, c, h, w) = x[0].shape
        y_h = conv.get_conv_outsize(h, self.kh, self.sy, self.ph)
        y_w = conv.get_conv_outsize(w, self.kw, self.sx, self.pw)
        y = cuda.cupy.empty((n, c, y_h, y_w), dtype=x[0].dtype)
        coeff = 1.0 / (self.kh * self.kw)
        kern = cuda.elementwise('raw T in, int32 h, int32 w,int32 out_h, int32 out_w, int32 kh, int32 kw,int32 sy, int32 sx, int32 ph, int32 pw, T coeff', 'T out', '\n            int c0    = i / (out_h * out_w);\n            int out_y = i / out_w % out_h;\n            int out_x = i % out_w;\n            int in_y_0 = max(0, out_y * sy - ph);\n            int in_y_1 = min(h, out_y * sy + kh - ph);\n            int in_x_0 = max(0, out_x * sx - pw);\n            int in_x_1 = min(w, out_x * sx + kw - pw);\n\n            T val = 0;\n            for (int y = in_y_0; y < in_y_1; ++y) {\n              int offset_y = w * (y + h * c0);\n              for (int x = in_x_0; x < in_x_1; ++x) {\n                val = val + in[x + offset_y];\n              }\n            }\n            out = val * coeff;\n            ', 'avg_pool_fwd')
        kern(x[0].reduced_view(), h, w, y_h, y_w, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, coeff, y)
        return (y,)

    def backward(self, indexes, gy):
        if False:
            while True:
                i = 10
        return AveragePooling2DGrad(self).apply(gy)

    def _get_pool_mode(self):
        if False:
            i = 10
            return i + 15
        return cuda.cuda.cudnn.CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING

class AveragePooling2DGrad(function_node.FunctionNode):

    def __init__(self, apool2d):
        if False:
            print('Hello World!')
        self.kh = apool2d.kh
        self.kw = apool2d.kw
        self.sy = apool2d.sy
        self.sx = apool2d.sx
        self.ph = apool2d.ph
        self.pw = apool2d.pw
        self._used_cudnn = apool2d._used_cudnn
        if not self._used_cudnn:
            self._in_shape = apool2d._in_shape
            self._in_dtype = apool2d._in_dtype
        self.apool2d = apool2d

    def forward_cpu(self, gy):
        if False:
            print('Hello World!')
        if intel64.should_use_ideep('>=auto') and intel64.inputs_all_ready(gy):
            return self._forward_ideep(gy)
        (h, w) = self._in_shape[2:]
        gcol = numpy.tile(gy[0][:, :, None, None], (1, 1, self.kh, self.kw, 1, 1))
        gx = conv.col2im_cpu(gcol, self.sy, self.sx, self.ph, self.pw, h, w)
        gx /= self.kh * self.kw
        return (gx,)

    def _forward_ideep(self, gy):
        if False:
            for i in range(10):
                print('nop')
        (n, c, h, w) = self._in_shape
        (y_h, y_w) = gy[0].shape[2:]
        (x,) = self.apool2d.get_retained_inputs()
        pd = self.sy * (y_h - 1) + self.kh - h - self.ph
        pr = self.sx * (y_w - 1) + self.kw - w - self.pw
        pp = intel64.ideep.pooling2DParam(self._in_shape, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, pd, pr, intel64.ideep.pooling2DParam.pooling_avg_include_padding)
        gx = intel64.ideep.pooling2D.Backward(intel64.ideep.array(x.data), intel64.ideep.array(gy[0]), None, pp)
        return (gx,)

    def forward_gpu(self, gy):
        if False:
            for i in range(10):
                print('nop')
        if self._used_cudnn:
            (x,) = self.apool2d.get_retained_inputs()
            return self.apool2d.backward_gpu((x.data,), gy)
        (n, c, h, w) = self._in_shape
        (y_h, y_w) = gy[0].shape[2:]
        gx = cuda.cupy.empty(self._in_shape, self._in_dtype)
        coeff = 1.0 / (self.kh * self.kw)
        cuda.elementwise('raw T gy, int32 h, int32 w,int32 out_h, int32 out_w, int32 kh, int32 kw,int32 sy, int32 sx, int32 ph, int32 pw, T coeff', 'T gx', '\n               int c0 = i / (h * w);\n               int y  = i / w % h + ph;\n               int x  = i % w + pw;\n               int out_y_0 = max(0,     (y - kh + sy) / sy);\n               int out_y_1 = min(out_h, (y      + sy) / sy);\n               int out_x_0 = max(0,     (x - kw + sx) / sx);\n               int out_x_1 = min(out_w, (x      + sx) / sx);\n               int hc0  = out_h * c0;\n\n               T val = 0;\n               for (int out_y = out_y_0; out_y < out_y_1; ++out_y) {\n                 for (int out_x = out_x_0; out_x < out_x_1; ++out_x) {\n                   val = val + gy[out_x + out_w * (out_y + hc0)];\n                 }\n               }\n               gx = val * coeff;\n            ', 'avg_pool_bwd')(gy[0].reduced_view(), h, w, y_h, y_w, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, coeff, gx)
        return (gx,)

    def backward(self, indexes, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        return AveragePooling2D((self.kh, self.kw), (self.sy, self.sx), (self.ph, self.pw), False).apply(grad_outputs)

def average_pooling_2d(x, ksize, stride=None, pad=0):
    if False:
        print('Hello World!')
    'Spatial average pooling function.\n\n    This function acts similarly to :func:`~chainer.functions.convolution_2d`,\n    but it computes the average of input spatial patch for each channel without\n    any parameter instead of computing the inner products.\n\n    Args:\n        x (~chainer.Variable): Input variable.\n        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and\n            ``ksize=(k, k)`` are equivalent.\n        stride (int or pair of ints or None): Stride of pooling applications.\n            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is\n            specified, then it uses same stride as the pooling window size.\n        pad (int or pair of ints): Spatial padding width for the input array.\n            ``pad=p`` and ``pad=(p, p)`` are equivalent.\n\n    Returns:\n        ~chainer.Variable: Output variable.\n\n    .. note::\n\n       This function currently does not support ``cover_all`` mode as\n       :func:`max_pooling_2d`. Average pooling runs in non-cover-all mode.\n\n    .. note::\n\n       The values in the padded region is treated as 0, leading the averages\n       biased towards zero.\n       To obtain unbiased averages, use :func:`average_pooling_nd` with\n       ``pad_value=None``.\n\n    '
    if backend.get_array_module(x) is chainerx:
        return average_pooling_nd.average_pooling_nd(x, ksize, stride, pad)
    return AveragePooling2D(ksize, stride, pad, False).apply((x,))[0]