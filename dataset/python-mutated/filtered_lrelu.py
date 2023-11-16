import os
import warnings
import numpy as np
import torch
from .. import custom_ops, misc
from . import bias_act, upfirdn2d
_plugin = None

def _init():
    if False:
        i = 10
        return i + 15
    global _plugin
    if _plugin is None:
        _plugin = custom_ops.get_plugin(module_name='filtered_lrelu_plugin', sources=['filtered_lrelu.cpp', 'filtered_lrelu_wr.cu', 'filtered_lrelu_rd.cu', 'filtered_lrelu_ns.cu'], headers=['filtered_lrelu.h', 'filtered_lrelu.cu'], source_dir=os.path.dirname(__file__), extra_cuda_cflags=['--use_fast_math'])
    return True

def _get_filter_size(f):
    if False:
        i = 10
        return i + 15
    if f is None:
        return (1, 1)
    assert isinstance(f, torch.Tensor)
    assert 1 <= f.ndim <= 2
    return (f.shape[-1], f.shape[0])

def _parse_padding(padding):
    if False:
        print('Hello World!')
    if isinstance(padding, int):
        padding = [padding, padding]
    assert isinstance(padding, (list, tuple))
    assert all((isinstance(x, (int, np.integer)) for x in padding))
    padding = [int(x) for x in padding]
    if len(padding) == 2:
        (px, py) = padding
        padding = [px, px, py, py]
    (px0, px1, py0, py1) = padding
    return (px0, px1, py0, py1)

def filtered_lrelu(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False, impl='cuda'):
    if False:
        for i in range(10):
            print('nop')
    "Filtered leaky ReLU for a batch of 2D images.\n\n    Performs the following sequence of operations for each channel:\n\n    1. Add channel-specific bias if provided (`b`).\n\n    2. Upsample the image by inserting N-1 zeros after each pixel (`up`).\n\n    3. Pad the image with the specified number of zeros on each side (`padding`).\n       Negative padding corresponds to cropping the image.\n\n    4. Convolve the image with the specified upsampling FIR filter (`fu`), shrinking it\n       so that the footprint of all output pixels lies within the input image.\n\n    5. Multiply each value by the provided gain factor (`gain`).\n\n    6. Apply leaky ReLU activation function to each value.\n\n    7. Clamp each value between -clamp and +clamp, if `clamp` parameter is provided.\n\n    8. Convolve the image with the specified downsampling FIR filter (`fd`), shrinking\n       it so that the footprint of all output pixels lies within the input image.\n\n    9. Downsample the image by keeping every Nth pixel (`down`).\n\n    The fused op is considerably more efficient than performing the same calculation\n    using standard PyTorch ops. It supports gradients of arbitrary order.\n\n    Args:\n        x:           Float32/float16/float64 input tensor of the shape\n                     `[batch_size, num_channels, in_height, in_width]`.\n        fu:          Float32 upsampling FIR filter of the shape\n                     `[filter_height, filter_width]` (non-separable),\n                     `[filter_taps]` (separable), or\n                     `None` (identity).\n        fd:          Float32 downsampling FIR filter of the shape\n                     `[filter_height, filter_width]` (non-separable),\n                     `[filter_taps]` (separable), or\n                     `None` (identity).\n        b:           Bias vector, or `None` to disable. Must be a 1D tensor of the same type\n                     as `x`. The length of vector must must match the channel dimension of `x`.\n        up:          Integer upsampling factor (default: 1).\n        down:        Integer downsampling factor. (default: 1).\n        padding:     Padding with respect to the upsampled image. Can be a single number\n                     or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`\n                     (default: 0).\n        gain:        Overall scaling factor for signal magnitude (default: sqrt(2)).\n        slope:       Slope on the negative side of leaky ReLU (default: 0.2).\n        clamp:       Maximum magnitude for leaky ReLU output (default: None).\n        flip_filter: False = convolution, True = correlation (default: False).\n        impl:        Implementation to use. Can be `'ref'` or `'cuda'` (default: `'cuda'`).\n\n    Returns:\n        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.\n    "
    assert isinstance(x, torch.Tensor)
    assert impl in ['ref', 'cuda']
    if impl == 'cuda' and x.device.type == 'cuda' and _init():
        return _filtered_lrelu_cuda(up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter).apply(x, fu, fd, b, None, 0, 0)
    return _filtered_lrelu_ref(x, fu=fu, fd=fd, b=b, up=up, down=down, padding=padding, gain=gain, slope=slope, clamp=clamp, flip_filter=flip_filter)

@misc.profiled_function
def _filtered_lrelu_ref(x, fu=None, fd=None, b=None, up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    if False:
        return 10
    'Slow and memory-inefficient reference implementation of `filtered_lrelu()` using\n    existing `upfirdn2n()` and `bias_act()` ops.\n    '
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    (fu_w, fu_h) = _get_filter_size(fu)
    (fd_w, fd_h) = _get_filter_size(fd)
    if b is not None:
        assert isinstance(b, torch.Tensor) and b.dtype == x.dtype
        misc.assert_shape(b, [x.shape[1]])
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    (px0, px1, py0, py1) = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    assert slope == float(slope) and slope >= 0
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    (batch_size, channels, in_h, in_w) = x.shape
    in_dtype = x.dtype
    temp_w = in_w * up + (px0 + px1) - (fu_w - 1) - (fd_w - 1) + (down - 1)
    out_w = temp_w // down
    temp_h = in_h * up + (py0 + py1) - (fu_h - 1) - (fd_h - 1) + (down - 1)
    out_h = temp_h // down
    x = bias_act.bias_act(x=x, b=b)
    x = upfirdn2d.upfirdn2d(x=x, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
    x = bias_act.bias_act(x=x, act='lrelu', alpha=slope, gain=gain, clamp=clamp)
    x = upfirdn2d.upfirdn2d(x=x, f=fd, down=down, flip_filter=flip_filter)
    misc.assert_shape(x, [batch_size, channels, out_h, out_w])
    assert x.dtype == in_dtype
    return x
_filtered_lrelu_cuda_cache = dict()

def _filtered_lrelu_cuda(up=1, down=1, padding=0, gain=np.sqrt(2), slope=0.2, clamp=None, flip_filter=False):
    if False:
        print('Hello World!')
    'Fast CUDA implementation of `filtered_lrelu()` using custom ops.\n    '
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    (px0, px1, py0, py1) = _parse_padding(padding)
    assert gain == float(gain) and gain > 0
    gain = float(gain)
    assert slope == float(slope) and slope >= 0
    slope = float(slope)
    assert clamp is None or (clamp == float(clamp) and clamp >= 0)
    clamp = float(clamp if clamp is not None else 'inf')
    key = (up, down, px0, px1, py0, py1, gain, slope, clamp, flip_filter)
    if key in _filtered_lrelu_cuda_cache:
        return _filtered_lrelu_cuda_cache[key]

    class FilteredLReluCuda(torch.autograd.Function):

        @staticmethod
        def forward(ctx, x, fu, fd, b, si, sx, sy):
            if False:
                return 10
            assert isinstance(x, torch.Tensor) and x.ndim == 4
            if fu is None:
                fu = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            if fd is None:
                fd = torch.ones([1, 1], dtype=torch.float32, device=x.device)
            assert 1 <= fu.ndim <= 2
            assert 1 <= fd.ndim <= 2
            if up == 1 and fu.ndim == 1 and (fu.shape[0] == 1):
                fu = fu.square()[None]
            if down == 1 and fd.ndim == 1 and (fd.shape[0] == 1):
                fd = fd.square()[None]
            if si is None:
                si = torch.empty([0])
            if b is None:
                b = torch.zeros([x.shape[1]], dtype=x.dtype, device=x.device)
            write_signs = si.numel() == 0 and (x.requires_grad or b.requires_grad)
            strides = [x.stride(i) for i in range(x.ndim) if x.size(i) > 1]
            if any((a < b for (a, b) in zip(strides[:-1], strides[1:]))):
                warnings.warn('low-performance memory layout detected in filtered_lrelu input', RuntimeWarning)
            if x.dtype in [torch.float16, torch.float32]:
                if torch.cuda.current_stream(x.device) != torch.cuda.default_stream(x.device):
                    warnings.warn('filtered_lrelu called with non-default cuda stream but concurrent execution is not supported', RuntimeWarning)
                (y, so, return_code) = _plugin.filtered_lrelu(x, fu, fd, b, si, up, down, px0, px1, py0, py1, sx, sy, gain, slope, clamp, flip_filter, write_signs)
            else:
                return_code = -1
            if return_code < 0:
                warnings.warn('filtered_lrelu called with parameters that have no optimized CUDA kernel, using generic fallback', RuntimeWarning)
                y = x.add(b.unsqueeze(-1).unsqueeze(-1))
                y = upfirdn2d.upfirdn2d(x=y, f=fu, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
                so = _plugin.filtered_lrelu_act_(y, si, sx, sy, gain, slope, clamp, write_signs)
                y = upfirdn2d.upfirdn2d(x=y, f=fd, down=down, flip_filter=flip_filter)
            ctx.save_for_backward(fu, fd, si if si.numel() else so)
            ctx.x_shape = x.shape
            ctx.y_shape = y.shape
            ctx.s_ofs = (sx, sy)
            return y

        @staticmethod
        def backward(ctx, dy):
            if False:
                for i in range(10):
                    print('nop')
            (fu, fd, si) = ctx.saved_tensors
            (_, _, xh, xw) = ctx.x_shape
            (_, _, yh, yw) = ctx.y_shape
            (sx, sy) = ctx.s_ofs
            dx = None
            dfu = None
            assert not ctx.needs_input_grad[1]
            dfd = None
            assert not ctx.needs_input_grad[2]
            db = None
            dsi = None
            assert not ctx.needs_input_grad[4]
            dsx = None
            assert not ctx.needs_input_grad[5]
            dsy = None
            assert not ctx.needs_input_grad[6]
            if ctx.needs_input_grad[0] or ctx.needs_input_grad[3]:
                pp = [fu.shape[-1] - 1 + (fd.shape[-1] - 1) - px0, xw * up - yw * down + px0 - (up - 1), fu.shape[0] - 1 + (fd.shape[0] - 1) - py0, xh * up - yh * down + py0 - (up - 1)]
                gg = gain * up ** 2 / down ** 2
                ff = not flip_filter
                sx = sx - (fu.shape[-1] - 1) + px0
                sy = sy - (fu.shape[0] - 1) + py0
                dx = _filtered_lrelu_cuda(up=down, down=up, padding=pp, gain=gg, slope=slope, clamp=None, flip_filter=ff).apply(dy, fd, fu, None, si, sx, sy)
            if ctx.needs_input_grad[3]:
                db = dx.sum([0, 2, 3])
            return (dx, dfu, dfd, db, dsi, dsx, dsy)
    _filtered_lrelu_cuda_cache[key] = FilteredLReluCuda
    return FilteredLReluCuda