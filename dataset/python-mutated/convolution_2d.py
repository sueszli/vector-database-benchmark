import numpy
import chainer
from chainer import backend
from chainer.backends import cuda
from chainer.backends import intel64
from chainer import configuration
from chainer import function_node
import chainer.functions
from chainer import memory_layouts
from chainer.utils import argument
from chainer.utils import conv
from chainer.utils import type_check
import chainerx
if cuda.cudnn_enabled:
    _cudnn_version = cuda.cuda.cudnn.getVersion()

def _pair(x):
    if False:
        i = 10
        return i + 15
    if hasattr(x, '__getitem__'):
        return x
    return (x, x)

def _matmul(a, b):
    if False:
        for i in range(10):
            print('nop')
    xp = backend.get_array_module(a)
    if not hasattr(xp, 'matmul'):
        return xp.einsum('ijl,ilk->ijk', a, b)
    return xp.matmul(a, b)

class Convolution2DFunction(function_node.FunctionNode):
    _use_ideep = False

    def __init__(self, stride=1, pad=0, cover_all=False, **kwargs):
        if False:
            while True:
                i = 10
        (dilate, groups, cudnn_fast) = argument.parse_kwargs(kwargs, ('dilate', 1), ('groups', 1), ('cudnn_fast', False), deterministic="deterministic argument is not supported anymore. Use chainer.using_config('cudnn_deterministic', value) context where value is either `True` or `False`.", requires_x_grad='requires_x_grad argument is not supported anymore. Just remove the argument. Note that whether to compute the gradient w.r.t. x is automatically decided during backpropagation.')
        (self.sy, self.sx) = _pair(stride)
        (self.ph, self.pw) = _pair(pad)
        self.cover_all = cover_all
        (self.dy, self.dx) = _pair(dilate)
        self.groups = groups
        self.cudnn_fast = cudnn_fast
        if self.dx < 1 or self.dy < 1:
            raise ValueError('Dilate should be positive, but {} is supplied.'.format(dilate))

    def check_type_forward(self, in_types):
        if False:
            i = 10
            return i + 15
        n_in = in_types.size()
        type_check.expect(2 <= n_in, n_in <= 3)
        x_type = in_types[0]
        w_type = in_types[1]
        type_check.expect(x_type.dtype.kind == 'f', w_type.dtype.kind == 'f', x_type.ndim == 4, w_type.ndim == 4, x_type.shape[1] == w_type.shape[1] * self.groups)
        if type_check.eval(n_in) == 3:
            b_type = in_types[2]
            type_check.expect(b_type.dtype == x_type.dtype, b_type.ndim == 1, b_type.shape[0] == w_type.shape[0])

    def check_layout_forward(self, inputs):
        if False:
            i = 10
            return i + 15
        input_layouts = self.input_layouts
        n = len(inputs)
        layouts = ((memory_layouts.CUDNN_CHANNEL_FIRST_X, memory_layouts.CUDNN_CHANNEL_LAST_X), (memory_layouts.CUDNN_CHANNEL_FIRST_W, memory_layouts.CUDNN_CHANNEL_LAST_W), (None,))
        for (i, (input_layout, expected_layouts)) in enumerate(zip(input_layouts, layouts[:n])):
            if input_layout not in expected_layouts:
                raise RuntimeError('Invalid layout for input {}: {}'.format(i, input_layout))

    def _get_out_size(self, x_shape, w_shape):
        if False:
            i = 10
            return i + 15
        (_, _, kh, kw) = w_shape
        (_, _, h, w) = x_shape
        out_h = conv.get_conv_outsize(h, kh, self.sy, self.ph, cover_all=self.cover_all, d=self.dy)
        if out_h <= 0:
            raise RuntimeError('Height in the output should be positive.')
        out_w = conv.get_conv_outsize(w, kw, self.sx, self.pw, cover_all=self.cover_all, d=self.dx)
        if out_w <= 0:
            raise RuntimeError('Width in the output should be positive.')
        return (out_h, out_w)

    def _check_input_layouts_all_standard(self):
        if False:
            i = 10
            return i + 15
        if not all([layout is None for layout in self.input_layouts]):
            raise RuntimeError('Non-standard memory layouts are only supported with cupy arrays in {}. Input layouts: {}'.format(self.label, self.input_layouts))

    def forward_chainerx(self, inputs):
        if False:
            print('Hello World!')
        if any([arr.dtype != inputs[0].dtype for arr in inputs[1:]]):
            return chainer.Fallback
        if self.dy > 1 or self.dx > 1:
            return chainer.Fallback
        if self.groups > 1:
            return chainer.Fallback
        if inputs[0].device.backend.name == 'cuda' and self.cover_all:
            return chainer.Fallback
        return (chainerx.conv(*inputs, stride=(self.sy, self.sx), pad=(self.ph, self.pw), cover_all=self.cover_all),)

    def forward_cpu(self, inputs):
        if False:
            i = 10
            return i + 15
        if self.cudnn_fast:
            raise RuntimeError("'cudnn_fast' can't be used in the CPU backend")
        self._check_input_layouts_all_standard()
        self.retain_inputs((0, 1))
        if len(inputs) == 2:
            ((x, W), b) = (inputs, None)
        else:
            (x, W, b) = inputs
        if intel64.should_use_ideep('>=auto') and intel64.inputs_all_ready(inputs):
            self._use_ideep = True
        if self.groups > 1:
            return self._forward_grouped_convolution(x, W, b)
        else:
            return self._forward_cpu_core(x, W, b)

    def _forward_cpu_core(self, x, W, b):
        if False:
            print('Hello World!')
        if self._use_ideep:
            return self._forward_ideep(x, W, b)
        (kh, kw) = W.shape[2:]
        col = conv.im2col_cpu(x, kh, kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = numpy.tensordot(col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        y = numpy.rollaxis(y, 3, 1)
        return (y,)

    def _forward_ideep(self, x, W, b):
        if False:
            for i in range(10):
                print('nop')
        (out_c, input_c, kh, kw) = W.shape
        (n, c, h, w) = x.shape
        (out_h, out_w) = self._get_out_size(x.shape, W.shape)
        pd = self.sy * (out_h - 1) + (kh + (kh - 1) * (self.dy - 1)) - h - self.ph
        pr = self.sx * (out_w - 1) + (kw + (kw - 1) * (self.dx - 1)) - w - self.pw
        param = intel64.ideep.convolution2DParam((n, out_c, out_h, out_w), self.dy, self.dx, self.sy, self.sx, self.ph, self.pw, pd, pr)
        y = intel64.ideep.convolution2D.Forward(intel64.ideep.array(x), intel64.ideep.array(W), intel64.ideep.array(b) if b is not None else None, param)
        return (y,)

    def forward_gpu(self, inputs):
        if False:
            for i in range(10):
                print('nop')
        self.retain_inputs((0, 1))
        if len(inputs) == 2:
            ((x, W), b) = (inputs, None)
            (x_layout, w_layout) = self.input_layouts
        else:
            (x, W, b) = inputs
            (x_layout, w_layout, _) = self.input_layouts
        x_shape = memory_layouts._transpose_shape(x.shape, x_layout, None)
        w_shape = memory_layouts._transpose_shape(W.shape, w_layout, None)
        (n, _, h, w) = x_shape
        (out_c, _, kh, kw) = w_shape
        (out_h, out_w) = self._get_out_size(x_shape, w_shape)
        y_raw_shape = memory_layouts._transpose_shape((n, out_c, out_h, out_w), None, x_layout)
        y = cuda.cupy.empty(y_raw_shape, dtype=x.dtype)
        use_cudnn = chainer.should_use_cudnn('>=auto') and (not self.cover_all) and (x.dtype == W.dtype) and (self.dy == 1 and self.dx == 1 or _cudnn_version >= 6000) and (self.groups <= 1 or _cudnn_version >= 7000)
        if self.cudnn_fast and (not use_cudnn):
            raise RuntimeError("'cudnn_fast' requires cuDNN to work")
        if use_cudnn:
            return self._forward_cudnn(x, W, b, y, (x_layout, w_layout))
        elif self.groups > 1:
            return self._forward_grouped_convolution(x, W, b)
        else:
            return self._forward_gpu_core(x, W, b)

    def _forward_gpu_core(self, x, W, b):
        if False:
            print('Hello World!')
        (kh, kw) = W.shape[2:]
        col = conv.im2col_gpu(x, kh, kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        y = cuda.cupy.tensordot(col, W, ((1, 2, 3), (1, 2, 3))).astype(x.dtype, copy=False)
        if b is not None:
            y += b
        y = cuda.cupy.rollaxis(y, 3, 1)
        return (y,)

    def _forward_grouped_convolution(self, x, W, b):
        if False:
            while True:
                i = 10
        G = self.groups
        (N, iC, iH, iW) = x.shape
        (oC, _, kH, kW) = W.shape
        iCg = iC // G
        oCg = oC // G
        x = conv.im2col(x, kH, kW, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        (oH, oW) = x.shape[-2:]
        x = x.transpose(1, 2, 3, 0, 4, 5)
        x = x.reshape(G, iCg * kH * kW, N * oH * oW)
        W = W.reshape(G, oCg, iCg * kH * kW)
        y = _matmul(W, x).astype(x.dtype, copy=False)
        y = y.reshape(oC, N, oH, oW)
        y = y.transpose(1, 0, 2, 3)
        if b is not None:
            y += b.reshape(1, b.size, 1, 1)
        return (y,)

    def _forward_cudnn(self, x, W, b, y, input_layouts):
        if False:
            while True:
                i = 10
        (x_layout, w_layout) = input_layouts
        self.output_layouts = (x_layout,)
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cudnn_x_layout = cuda._get_cudnn_tensor_layout_x(x_layout)
        cudnn_w_layout = cuda._get_cudnn_tensor_layout_w(w_layout)
        cuda.cudnn.convolution_forward(x, W, b, y, pad, stride, dilation, self.groups, auto_tune=auto_tune, tensor_core=tensor_core, d_layout=cudnn_x_layout, w_layout=cudnn_w_layout)
        return (y,)

    def backward(self, indexes, grad_outputs):
        if False:
            i = 10
            return i + 15
        (x, W) = self.get_retained_inputs()
        if len(self.input_layouts) == 2:
            (x_layout, _) = self.input_layouts
        else:
            (x_layout, _, _) = self.input_layouts
        (gy,) = grad_outputs
        ret = []
        if 0 in indexes:
            (_, _, xh, xw) = x.shape
            gx = chainer.functions.deconvolution_2d(gy, W, stride=(self.sy, self.sx), pad=(self.ph, self.pw), outsize=(xh, xw), dilate=(self.dy, self.dx), groups=self.groups)
            assert gx.shape == x.shape
            ret.append(gx)
        if 1 in indexes:
            (gW,) = Convolution2DGradW(self, W.shape, W.dtype, W.layout).apply((x, gy))
            ret.append(gW)
        if 2 in indexes:
            axis = (0, 2, 3)
            inv_trans = memory_layouts._get_layout_transpose_axes(gy.ndim, None, x_layout)
            if inv_trans is None:
                raw_axis = axis
            else:
                raw_axis = tuple([inv_trans[i] for i in axis])
            gb = chainer.functions.sum(gy, axis=raw_axis)
            ret.append(gb)
        return ret

class Convolution2DGradW(function_node.FunctionNode):

    def __init__(self, conv2d, w_shape, w_dtype, w_layout):
        if False:
            print('Hello World!')
        (self.kh, self.kw) = w_shape[2:]
        self.sy = conv2d.sy
        self.sx = conv2d.sx
        self.ph = conv2d.ph
        self.pw = conv2d.pw
        self.dy = conv2d.dy
        self.dx = conv2d.dx
        self.cover_all = conv2d.cover_all
        self.W_shape = w_shape
        self.W_dtype = w_dtype
        self.w_layout = w_layout
        self.groups = conv2d.groups
        self._use_ideep = conv2d._use_ideep

    def check_layout_forward(self, inputs):
        if False:
            while True:
                i = 10
        pass

    def forward_cpu(self, inputs):
        if False:
            print('Hello World!')
        self.retain_inputs((0, 1))
        (x, gy) = inputs
        if self.groups > 1:
            return self._forward_grouped_convolution(x, gy)
        else:
            return self._forward_cpu_core(x, gy)

    def _forward_cpu_core(self, x, gy):
        if False:
            return 10
        if self._use_ideep:
            return self._forward_ideep(x, gy)
        if not (gy.flags.c_contiguous or gy.flags.f_contiguous) and 1 in gy.shape:
            gy = numpy.ascontiguousarray(gy)
        col = conv.im2col_cpu(x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        gW = numpy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))).astype(self.W_dtype, copy=False)
        return (gW,)

    def _forward_ideep(self, x, gy):
        if False:
            print('Hello World!')
        (n, input_c, h, w) = x.shape
        (n, out_c, out_h, out_w) = gy.shape
        pd = self.sy * (out_h - 1) + (self.kh + (self.kh - 1) * (self.dy - 1)) - h - self.ph
        pr = self.sx * (out_w - 1) + (self.kw + (self.kw - 1) * (self.dx - 1)) - w - self.pw
        param = intel64.ideep.convolution2DParam((out_c, input_c, self.kh, self.kw), self.dy, self.dx, self.sy, self.sx, self.ph, self.pw, pd, pr)
        gW = intel64.ideep.convolution2D.BackwardWeights(intel64.ideep.array(x), intel64.ideep.array(gy), param)
        return (gW,)

    def forward_gpu(self, inputs):
        if False:
            return 10
        self.retain_inputs((0, 1))
        (x, gy) = inputs
        use_cudnn = chainer.should_use_cudnn('>=auto') and (not self.cover_all) and (x.dtype == self.W_dtype) and (self.dy == 1 and self.dx == 1 or (_cudnn_version >= 6000 and (not configuration.config.cudnn_deterministic))) and (self.groups <= 1 or _cudnn_version >= 7000)
        if use_cudnn:
            return self._forward_cudnn(x, gy)
        elif self.groups > 1:
            return self._forward_grouped_convolution(x, gy)
        else:
            return self._forward_gpu_core(x, gy)

    def _forward_gpu_core(self, x, gy):
        if False:
            for i in range(10):
                print('nop')
        col = conv.im2col_gpu(x, self.kh, self.kw, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        gW = cuda.cupy.tensordot(gy, col, ((0, 2, 3), (0, 4, 5))).astype(self.W_dtype, copy=False)
        return (gW,)

    def _forward_grouped_convolution(self, x, gy):
        if False:
            for i in range(10):
                print('nop')
        G = self.groups
        (N, iC, iH, iW) = x.shape
        (_, oC, oH, oW) = gy.shape
        kH = self.kh
        kW = self.kw
        iCg = iC // G
        oCg = oC // G
        x = conv.im2col(x, kH, kW, self.sy, self.sx, self.ph, self.pw, cover_all=self.cover_all, dy=self.dy, dx=self.dx)
        x = x.transpose(1, 2, 3, 0, 4, 5)
        x = x.reshape(G, iCg * kH * kW, N * oH * oW)
        x = x.transpose(0, 2, 1)
        gy = gy.transpose(1, 0, 2, 3)
        gy = gy.reshape(G, oCg, N * oH * oW)
        gW = _matmul(gy, x).astype(self.W_dtype, copy=False)
        gW = gW.reshape(oC, iCg, kH, kW)
        return (gW,)

    def _forward_cudnn(self, x, gy):
        if False:
            print('Hello World!')
        (x_layout, gy_layout) = self.input_layouts
        w_layout = self.w_layout
        w_raw_shape = memory_layouts._transpose_shape(self.W_shape, None, w_layout)
        gW = cuda.cupy.empty(w_raw_shape, dtype=self.W_dtype)
        pad = (self.ph, self.pw)
        stride = (self.sy, self.sx)
        dilation = (self.dy, self.dx)
        deterministic = configuration.config.cudnn_deterministic
        auto_tune = configuration.config.autotune
        tensor_core = configuration.config.use_cudnn_tensor_core
        cudnn_x_layout = cuda._get_cudnn_tensor_layout_x(x_layout)
        cudnn_w_layout = cuda._get_cudnn_tensor_layout_w(w_layout)
        cuda.cudnn.convolution_backward_filter(x, gy, gW, pad, stride, dilation, self.groups, deterministic=deterministic, auto_tune=auto_tune, tensor_core=tensor_core, d_layout=cudnn_x_layout, w_layout=cudnn_w_layout)
        return (gW,)

    def backward(self, indexes, grad_outputs):
        if False:
            for i in range(10):
                print('nop')
        (x, gy) = self.get_retained_inputs()
        (ggW,) = grad_outputs
        ret = []
        if 0 in indexes:
            (xh, xw) = x.shape[2:]
            gx = chainer.functions.deconvolution_2d(gy, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw), outsize=(xh, xw), dilate=(self.dy, self.dx), groups=self.groups)
            ret.append(gx)
        if 1 in indexes:
            ggy = convolution_2d(x, ggW, stride=(self.sy, self.sx), pad=(self.ph, self.pw), cover_all=self.cover_all, dilate=(self.dy, self.dx), groups=self.groups)
            ret.append(ggy)
        return ret

def convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, **kwargs):
    if False:
        i = 10
        return i + 15
    "convolution_2d(x, W, b=None, stride=1, pad=0, cover_all=False, *, dilate=1, groups=1)\n\n    Two-dimensional convolution function.\n\n    This is an implementation of two-dimensional convolution in ConvNets.\n    It takes three variables: the input image ``x``, the filter weight ``W``,\n    and the bias vector ``b``.\n\n    Notation: here is a notation for dimensionalities.\n\n    - :math:`n` is the batch size.\n    - :math:`c_I` and :math:`c_O` are the number of the input and output\n      channels, respectively.\n    - :math:`h_I` and :math:`w_I` are the height and width of the input image,\n      respectively.\n    - :math:`h_K` and :math:`w_K` are the height and width of the filters,\n      respectively.\n    - :math:`h_P` and :math:`w_P` are the height and width of the spatial\n      padding size, respectively.\n\n    Then the ``Convolution2D`` function computes correlations between filters\n    and patches of size :math:`(h_K, w_K)` in ``x``.\n    Note that correlation here is equivalent to the inner product between\n    expanded vectors.\n    Patches are extracted at positions shifted by multiples of ``stride`` from\n    the first position ``(-h_P, -w_P)`` for each spatial axis.\n    The right-most (or bottom-most) patches do not run over the padded spatial\n    size.\n\n    Let :math:`(s_Y, s_X)` be the stride of filter application. Then, the\n    output size :math:`(h_O, w_O)` is determined by the following equations:\n\n    .. math::\n\n       h_O &= (h_I + 2h_P - h_K) / s_Y + 1,\\\\\n       w_O &= (w_I + 2w_P - w_K) / s_X + 1.\n\n    If ``cover_all`` option is ``True``, the filter will cover the all\n    spatial locations. So, if the last stride of filter does not cover the\n    end of spatial locations, an additional stride will be applied to the end\n    part of spatial locations. In this case, the output size :math:`(h_O, w_O)`\n    is determined by the following equations:\n\n    .. math::\n\n       h_O &= (h_I + 2h_P - h_K + s_Y - 1) / s_Y + 1,\\\\\n       w_O &= (w_I + 2w_P - w_K + s_X - 1) / s_X + 1.\n\n    If the bias vector is given, then it is added to all spatial locations of\n    the output of convolution.\n\n    The output of this function can be non-deterministic when it uses cuDNN.\n    If ``chainer.configuration.config.cudnn_deterministic`` is ``True`` and\n    cuDNN version is >= v3, it forces cuDNN to use a deterministic algorithm.\n\n    Convolution links can use a feature of cuDNN called autotuning, which\n    selects the most efficient CNN algorithm for images of fixed-size,\n    can provide a significant performance boost for fixed neural nets.\n    To enable, set `chainer.using_config('autotune', True)`\n\n    When the dilation factor is greater than one, cuDNN is not used unless\n    the version is 6.0 or higher.\n\n    Args:\n        x (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Input variable of shape :math:`(n, c_I, h_I, w_I)`.\n        W (:class:`~chainer.Variable` or :ref:`ndarray`):\n            Weight variable of shape :math:`(c_O, c_I, h_K, w_K)`.\n        b (None or :class:`~chainer.Variable` or :ref:`ndarray`):\n            Bias variable of length :math:`c_O` (optional).\n        stride (:class:`int` or pair of :class:`int` s):\n            Stride of filter applications. ``stride=s`` and ``stride=(s, s)``\n            are equivalent.\n        pad (:class:`int` or pair of :class:`int` s):\n            Spatial padding width for input arrays.\n            ``pad=p`` and ``pad=(p, p)`` are equivalent.\n        cover_all (:class:`bool`):\n            If ``True``, all spatial locations are convoluted into some output\n            pixels.\n        dilate (:class:`int` or pair of :class:`int` s):\n            Dilation factor of filter applications.\n            ``dilate=d`` and ``dilate=(d, d)`` are equivalent.\n        groups (:class:`int`): Number of groups of channels. If the number\n            is greater than 1, input tensor :math:`W` is divided into some\n            blocks by this value. For each tensor blocks, convolution\n            operation will be executed independently. Input channel size\n            :math:`c_I` and output channel size :math:`c_O` must be exactly\n            divisible by this value.\n\n    Returns:\n        ~chainer.Variable:\n            Output variable of shape :math:`(n, c_O, h_O, w_O)`.\n\n    .. seealso::\n\n        :class:`~chainer.links.Convolution2D` to manage the model parameters\n        ``W`` and ``b``.\n\n    .. admonition:: Example\n\n        >>> n = 10\n        >>> c_i, c_o = 3, 1\n        >>> h_i, w_i = 30, 40\n        >>> h_k, w_k = 10, 10\n        >>> h_p, w_p = 5, 5\n        >>> x = np.random.uniform(0, 1, (n, c_i, h_i, w_i)).astype(np.float32)\n        >>> x.shape\n        (10, 3, 30, 40)\n        >>> W = np.random.uniform(0, 1, (c_o, c_i, h_k, w_k)).astype(np.float32)\n        >>> W.shape\n        (1, 3, 10, 10)\n        >>> b = np.random.uniform(0, 1, (c_o,)).astype(np.float32)\n        >>> b.shape\n        (1,)\n        >>> s_y, s_x = 5, 7\n        >>> y = F.convolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p))\n        >>> y.shape\n        (10, 1, 7, 6)\n        >>> h_o = int((h_i + 2 * h_p - h_k) / s_y + 1)\n        >>> w_o = int((w_i + 2 * w_p - w_k) / s_x + 1)\n        >>> y.shape == (n, c_o, h_o, w_o)\n        True\n        >>> y = F.convolution_2d(x, W, b, stride=(s_y, s_x), pad=(h_p, w_p), cover_all=True)\n        >>> y.shape == (n, c_o, h_o, w_o + 1)\n        True\n\n    "
    (dilate, groups, cudnn_fast) = argument.parse_kwargs(kwargs, ('dilate', 1), ('groups', 1), ('cudnn_fast', False), deterministic="deterministic argument is not supported anymore. Use chainer.using_config('cudnn_deterministic', value) context where value is either `True` or `False`.")
    fnode = Convolution2DFunction(stride, pad, cover_all, dilate=dilate, groups=groups, cudnn_fast=cudnn_fast)
    if b is None:
        args = (x, W)
    else:
        args = (x, W, b)
    (y,) = fnode.apply(args)
    return y