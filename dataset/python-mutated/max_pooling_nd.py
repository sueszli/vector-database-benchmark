import functools
from operator import mul
import numpy
import six
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function_node
from chainer.functions.pooling import max_pooling_nd_kernel
from chainer.functions.pooling import pooling_nd
from chainer.utils import conv_nd
import chainerx
if cuda.cudnn_enabled:
    _cudnn_version = cuda.cuda.cudnn.getVersion()

class MaxPoolingND(pooling_nd._PoolingND):
    """Max pooling over a set of N-dimensional planes.

    .. warning::

        This feature is experimental. The interface can change in the future.

    """

    def __init__(self, ndim, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
        if False:
            return 10
        super(MaxPoolingND, self).__init__(ndim, ksize, stride=stride, pad=pad, cover_all=cover_all, return_indices=return_indices)

    def forward_chainerx(self, x):
        if False:
            print('Hello World!')
        ndim = self.ndim
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        cover_all = self.cover_all
        if self.return_indices:
            return chainer.Fallback
        if x[0].device.backend.name == 'cuda':
            if ndim not in [2, 3]:
                return chainer.Fallback
        y = chainerx.max_pool(x[0], ksize, stride, pad, cover_all)
        return (y,)

    def forward_cpu(self, x):
        if False:
            while True:
                i = 10
        if self.ndim == 2 and intel64.should_use_ideep('>=auto') and intel64.inputs_all_ready(x):
            return self._forward_2d_ideep(x)
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        cover_all = self.cover_all
        in_shape = x[0].shape
        in_dtype = x[0].dtype
        col = conv_nd.im2col_nd_cpu(x[0], ksize, stride, pad, pval=-float('inf'), cover_all=cover_all)
        (n, c) = col.shape[:2]
        mid = (len(col.shape) - 2) // 2 + 2
        ksize = col.shape[2:mid]
        outs = col.shape[mid:]
        col_shape = (n, c) + (functools.reduce(mul, ksize),) + outs
        col = col.reshape(col_shape)
        y = col.max(axis=2)
        self._in_shape = in_shape
        self._in_dtype = in_dtype
        self.indexes = col.argmax(axis=2)
        return (y,)

    def _forward_2d_ideep(self, x):
        if False:
            while True:
                i = 10
        assert self.ndim == 2
        (kh, kw) = self.ksize
        (sy, sx) = self.stride
        (ph, pw) = self.pad
        cover_all = self.cover_all
        self._in_shape = x[0].shape
        self._in_dtype = x[0].dtype
        self.retain_inputs((0,))
        (n, c, h, w) = x[0].shape
        y_h = conv_nd.get_conv_outsize(h, kh, sy, ph, cover_all)
        assert y_h > 0, 'Height in the output should be positive.'
        y_w = conv_nd.get_conv_outsize(w, kw, sx, pw, cover_all)
        assert y_w > 0, 'Width in the output should be positive.'
        pd = sy * (y_h - 1) + kh - h - ph
        pr = sx * (y_w - 1) + kw - w - pw
        pp = intel64.ideep.pooling2DParam((n, c, y_h, y_w), kh, kw, sy, sx, ph, pw, pd, pr, intel64.ideep.pooling2DParam.pooling_max)
        (y, indexes) = intel64.ideep.pooling2D.Forward(intel64.ideep.array(x[0]), pp)
        self.indexes = indexes
        return (y,)

    def forward_gpu(self, x):
        if False:
            for i in range(10):
                print('nop')
        if chainer.should_use_cudnn('>=auto') and 2 <= self.ndim <= 3:
            return self.forward_cudnn(x)
        ndim = self.ndim
        ksize = self.ksize
        stride = self.stride
        pad = self.pad
        cover_all = self.cover_all
        in_shape = x[0].shape
        in_dtype = x[0].dtype
        (n, c) = in_shape[:2]
        dims = in_shape[2:]
        ys = tuple((conv_nd.get_conv_outsize(d, k, s, p, cover_all) for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad)))
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x[0].dtype)
        indexes = cuda.cupy.empty(y_shape, dtype=numpy.int32)
        (in_params, out_params, operation, name) = max_pooling_nd_kernel.MaxPoolingNDKernelForward.generate(ndim)
        cuda.elementwise(in_params, out_params, operation, name)(x[0].reduced_view(), *dims + ys + ksize + stride + pad + (y, indexes))
        self._in_shape = in_shape
        self._in_dtype = in_dtype
        self.indexes = indexes
        return (y,)

    def backward(self, indexes, gy):
        if False:
            for i in range(10):
                print('nop')
        return MaxPoolingNDGrad(self).apply(gy)

    def get_cudnn_pool_mode(self):
        if False:
            print('Hello World!')
        if _cudnn_version >= 6000 and configuration.config.cudnn_deterministic:
            return cuda.cuda.cudnn.CUDNN_POOLING_MAX_DETERMINISTIC
        else:
            return cuda.cuda.cudnn.CUDNN_POOLING_MAX

class MaxPoolingNDGrad(function_node.FunctionNode):

    def __init__(self, func):
        if False:
            return 10
        self.func = func

    def forward_cpu(self, gy):
        if False:
            return 10
        func = self.func
        if func.ndim == 2 and intel64.should_use_ideep('>=auto') and intel64.inputs_all_ready(gy):
            return self._forward_2d_ideep(gy)
        ndim = func.ndim
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        in_shape = func._in_shape
        in_dtype = func._in_dtype
        indexes = func.indexes
        (n, c) = gy[0].shape[:2]
        outs = gy[0].shape[2:]
        dims = in_shape[2:]
        prod_outs = functools.reduce(mul, outs)
        prod_ksize = functools.reduce(mul, ksize)
        gcol = numpy.zeros(n * c * prod_outs * prod_ksize, dtype=in_dtype)
        indexes = indexes.flatten() + numpy.arange(0, indexes.size * prod_ksize, prod_ksize)
        gcol[indexes] = gy[0].ravel()
        gcol_shape = (n, c) + outs + ksize
        gcol = gcol.reshape(gcol_shape)
        for i in six.moves.range(ndim):
            gcol = numpy.swapaxes(gcol, 2 + i, ndim + 2 + i)
        gx = conv_nd.col2im_nd_cpu(gcol, stride, pad, dims)
        return (gx,)

    def _forward_2d_ideep(self, gy):
        if False:
            while True:
                i = 10
        func = self.func
        if not isinstance(func.indexes, intel64.ideep.mdarray):
            return self.forward_cpu(gy)
        (kh, kw) = func.ksize
        (sy, sx) = func.stride
        (ph, pw) = func.pad
        indexes = func.indexes
        in_shape = func._in_shape
        (n, c, h, w) = in_shape
        (y_h, y_w) = gy[0].shape[2:]
        x = func.get_retained_inputs()[0].array
        pd = sy * (y_h - 1) + kh - h - ph
        pr = sx * (y_w - 1) + kw - w - pw
        pp = intel64.ideep.pooling2DParam(func._in_shape, kh, kw, sy, sx, ph, pw, pd, pr, intel64.ideep.pooling2DParam.pooling_max)
        indexes = intel64.ideep.array(indexes)
        gx = intel64.ideep.pooling2D.Backward(intel64.ideep.array(x), intel64.ideep.array(gy[0]), indexes, pp)
        return (gx,)

    def forward_gpu(self, gy):
        if False:
            for i in range(10):
                print('nop')
        func = self.func
        if func.is_cudnn_used:
            return func.backward_cudnn(gy)
        ndim = func.ndim
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        in_shape = func._in_shape
        in_dtype = func._in_dtype
        indexes = backend.from_chx(func.indexes)
        (n, c) = in_shape[:2]
        dims = in_shape[2:]
        ys = gy[0].shape[2:]
        gx = cuda.cupy.empty(in_shape, in_dtype)
        (in_params, out_params, operation, name) = max_pooling_nd_kernel.MaxPoolingNDKernelBackward.generate(ndim)
        cuda.elementwise(in_params, out_params, operation, name)(gy[0].reduced_view(), indexes.reduced_view(), *dims + ys + ksize + stride + pad + (gx,))
        return (gx,)

    def backward(self, indexes, ggx):
        if False:
            i = 10
            return i + 15
        return MaxPoolingNDWithIndexes(self.func).apply(ggx)

class MaxPoolingNDWithIndexes(function_node.FunctionNode):

    def __init__(self, func):
        if False:
            i = 10
            return i + 15
        self.func = func

    def forward_cpu(self, x):
        if False:
            return 10
        func = self.func
        ndim = func.ndim
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        cover_all = func.cover_all
        indexes = backend.from_chx(func.indexes)
        col = conv_nd.im2col_nd_cpu(x[0], ksize, stride, pad, pval=-float('inf'), cover_all=cover_all)
        (n, c) = col.shape[:2]
        mid = (len(col.shape) - 2) // 2 + 2
        ksize = col.shape[2:mid]
        outs = col.shape[mid:]
        ksize_total = functools.reduce(mul, ksize)
        col_shape = (n, c) + (ksize_total,) + outs
        col = col.reshape(col_shape)
        col_indexes = (0, 1) + tuple(six.moves.range(3, 3 + ndim)) + (2,)
        col = col.transpose(col_indexes)
        col = col.reshape(-1, ksize_total)
        indexes = indexes.ravel()
        col = col[numpy.arange(len(indexes)), indexes]
        return (col.reshape((n, c) + outs),)

    def forward_gpu(self, inputs):
        if False:
            return 10
        func = self.func
        if func.is_cudnn_used:
            x = func.get_retained_inputs()[0].array
            return self._forward_gpu_compute_indexes_again((x, inputs[0]))
        ndim = func.ndim
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        cover_all = func.cover_all
        indexes = backend.from_chx(func.indexes)
        (x,) = inputs
        in_shape = x.shape
        in_dtype = x.dtype
        (n, c) = in_shape[:2]
        dims = in_shape[2:]
        ys = tuple((conv_nd.get_conv_outsize(d, k, s, p, cover_all) for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad)))
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)
        cls = max_pooling_nd_kernel.MaxPoolingNDKernelForwardWithIndexes
        (in_params, out_params, operation, name) = cls.generate(ndim)
        cuda.elementwise(in_params, out_params, operation, name)(x.reduced_view(), *dims + ys + ksize + stride + pad + (indexes.reduced_view(), y))
        self._in_shape = in_shape
        self._in_dtype = in_dtype
        return (y,)

    def _forward_gpu_compute_indexes_again(self, inputs):
        if False:
            return 10
        func = self.func
        ndim = func.ndim
        ksize = func.ksize
        stride = func.stride
        pad = func.pad
        cover_all = func.cover_all
        (x, ggx) = inputs
        in_shape = x.shape
        in_dtype = x.dtype
        (n, c) = in_shape[:2]
        dims = in_shape[2:]
        ys = tuple((conv_nd.get_conv_outsize(d, k, s, p, cover_all) for (d, k, s, p) in six.moves.zip(dims, ksize, stride, pad)))
        y_shape = (n, c) + ys
        y = cuda.cupy.empty(y_shape, dtype=x.dtype)
        cls = max_pooling_nd_kernel.MaxPoolingNDKernelForwardWithIndexes1
        (in_params, out_params, operation, name) = cls.generate(ndim)
        cuda.elementwise(in_params, out_params, operation, name)(x.reduced_view(), *dims + ys + ksize + stride + pad + (ggx.reduced_view(), y))
        self._in_shape = in_shape
        self._in_dtype = in_dtype
        return (y,)

def max_pooling_nd(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    if False:
        for i in range(10):
            print('nop')
    'N-dimensionally spatial max pooling function.\n\n    .. warning::\n\n        This feature is experimental. The interface can change in the future.\n\n    This function provides a N-dimensionally generalized version of\n    :func:`~chainer.functions.max_pooling_2d`. This acts similarly to\n    :func:`~chainer.functions.convolution_nd`, but it computes the maximum of\n    input spatial patch for each channel without any parameter instead of\n    computing the inner products.\n\n    Args:\n        x (~chainer.Variable): Input variable.\n        ksize (int or tuple of ints): Size of pooling window. ``ksize=k`` and\n            ``ksize=(k, k, ..., k)`` are equivalent.\n        stride (int or tuple of ints or None): Stride of pooling applications.\n            ``stride=s`` and ``stride=(s,s, ..., s)`` are equivalent. If\n            ``None`` is specified, then it uses same stride as the pooling\n            window size.\n        pad (int or tuple of ints): Spatial padding width for the input array.\n            ``pad=p`` and ``pad=(p, p, ..., p)`` are equivalent.\n        cover_all (bool): If ``True``, all spatial locations are pooled into\n            some output pixels. It may make the output size larger.\n        return_indices (bool): If ``True``, pooling indices array is returned\n            together with the output variable. The returned indices are\n            expected for use by :func:`chainer.functions.upsampling_nd`.\n            Note that cuDNN will not be used for this function if\n            ``return_indices`` is set to ``True``, as cuDNN does not return\n            indices information.\n\n    Returns:\n        ~chainer.Variable or tuple:\n            When ``return_indices`` is ``False`` (default), returns the output\n            variable.\n            When ``True``, returns the tuple of the output variable and\n            pooling indices (:ref:`ndarray`). Pooling indices will be on the\n            same device as the input.\n\n    '
    ndim = len(x.shape[2:])
    func = MaxPoolingND(ndim, ksize, stride, pad, cover_all, return_indices)
    if return_indices:
        with chainer.using_config('use_cudnn', 'never'):
            out = func.apply((x,))[0]
        return (out, func.indexes)
    return func.apply((x,))[0]

def max_pooling_1d(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    if False:
        print('Hello World!')
    '1-dimensional spatial max pooling function.\n\n    .. warning::\n\n        This feature is experimental. The interface can change in the future.\n\n    .. note::\n\n        This function calls :func:`~chainer.functions.max_pooling_nd`\n        internally, so see the details of the behavior in\n        the documentation of :func:`~chainer.functions.max_pooling_nd`.\n\n    '
    if len(x.shape[2:]) != 1:
        raise ValueError("The number of dimensions under channel dimension of the input 'x' should be 1. But the actual ndim was {}.".format(len(x.shape[2:])))
    return max_pooling_nd(x, ksize, stride, pad, cover_all, return_indices)

def max_pooling_2d(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    if False:
        return 10
    'Spatial max pooling function.\n\n    This function acts similarly to :func:`~chainer.functions.convolution_2d`,\n    but it computes the maximum of input spatial patch for each channel without\n    any parameter instead of computing the inner products.\n\n    Args:\n        x (~chainer.Variable): Input variable.\n        ksize (int or pair of ints): Size of pooling window. ``ksize=k`` and\n            ``ksize=(k, k)`` are equivalent.\n        stride (int or pair of ints or None): Stride of pooling applications.\n            ``stride=s`` and ``stride=(s, s)`` are equivalent. If ``None`` is\n            specified, then it uses same stride as the pooling window size.\n        pad (int or pair of ints): Spatial padding width for the input array.\n            ``pad=p`` and ``pad=(p, p)`` are equivalent.\n        cover_all (bool): If ``True``, all spatial locations are pooled into\n            some output pixels. It may make the output size larger.\n        return_indices (bool): If ``True``, pooling indices array is returned\n            together with the output variable. The returned indices are\n            expected for use by :func:`chainer.functions.upsampling_2d`.\n            Note that cuDNN will not be used for this function if\n            ``return_indices`` is set to ``True``, as cuDNN does not return\n            indices information.\n\n    Returns:\n        ~chainer.Variable or tuple:\n            When ``return_indices`` is ``False`` (default), returns the output\n            variable.\n            When ``True``, returns the tuple of the output variable and\n            pooling indices (:ref:`ndarray`). Pooling indices will be on the\n            same device as the input.\n\n    '
    if len(x.shape[2:]) != 2:
        raise ValueError("The number of dimensions under channel dimension of the input 'x' should be 2. But the actual ndim was {}.".format(len(x.shape[2:])))
    return max_pooling_nd(x, ksize, stride, pad, cover_all, return_indices)

def max_pooling_3d(x, ksize, stride=None, pad=0, cover_all=True, return_indices=False):
    if False:
        while True:
            i = 10
    '3-dimensional spatial max pooling function.\n\n    .. warning::\n\n        This feature is experimental. The interface can change in the future.\n\n    .. note::\n\n        This function calls :func:`~chainer.functions.max_pooling_nd`\n        internally, so see the details of the behavior in\n        the documentation of :func:`~chainer.functions.max_pooling_nd`.\n\n    '
    if len(x.shape[2:]) != 3:
        raise ValueError("The number of dimensions under channel dimension of the input 'x' should be 3. But the actual ndim was {}.".format(len(x.shape[2:])))
    return max_pooling_nd(x, ksize, stride, pad, cover_all, return_indices)