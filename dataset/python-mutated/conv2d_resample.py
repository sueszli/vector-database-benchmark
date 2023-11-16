"""2D convolution with optional up/downsampling."""
import torch
from .. import misc
from . import conv2d_gradfix, upfirdn2d
from .upfirdn2d import _get_filter_size, _parse_padding

def _get_weight_shape(w):
    if False:
        print('Hello World!')
    with misc.suppress_tracer_warnings():
        shape = [int(sz) for sz in w.shape]
    misc.assert_shape(w, shape)
    return shape

def _conv2d_wrapper(x, w, stride=1, padding=0, groups=1, transpose=False, flip_weight=True):
    if False:
        i = 10
        return i + 15
    'Wrapper for the underlying `conv2d()` and `conv_transpose2d()` implementations.\n    '
    (_out_channels, _in_channels_per_group, kh, kw) = _get_weight_shape(w)
    if not flip_weight and (kw > 1 or kh > 1):
        w = w.flip([2, 3])
    op = conv2d_gradfix.conv_transpose2d if transpose else conv2d_gradfix.conv2d
    return op(x, w, stride=stride, padding=padding, groups=groups)

@misc.profiled_function
def conv2d_resample(x, w, f=None, up=1, down=1, padding=0, groups=1, flip_weight=True, flip_filter=False):
    if False:
        return 10
    '2D convolution with optional up/downsampling.\n\n    Padding is performed only once at the beginning, not between the operations.\n\n    Args:\n        x:              Input tensor of shape\n                        `[batch_size, in_channels, in_height, in_width]`.\n        w:              Weight tensor of shape\n                        `[out_channels, in_channels//groups, kernel_height, kernel_width]`.\n        f:              Low-pass filter for up/downsampling. Must be prepared beforehand by\n                        calling upfirdn2d.setup_filter(). None = identity (default).\n        up:             Integer upsampling factor (default: 1).\n        down:           Integer downsampling factor (default: 1).\n        padding:        Padding with respect to the upsampled image. Can be a single number\n                        or a list/tuple `[x, y]` or `[x_before, x_after, y_before, y_after]`\n                        (default: 0).\n        groups:         Split input channels into N groups (default: 1).\n        flip_weight:    False = convolution, True = correlation (default: True).\n        flip_filter:    False = convolution, True = correlation (default: False).\n\n    Returns:\n        Tensor of the shape `[batch_size, num_channels, out_height, out_width]`.\n    '
    assert isinstance(x, torch.Tensor) and x.ndim == 4
    assert isinstance(w, torch.Tensor) and w.ndim == 4 and (w.dtype == x.dtype)
    assert f is None or (isinstance(f, torch.Tensor) and f.ndim in [1, 2] and (f.dtype == torch.float32))
    assert isinstance(up, int) and up >= 1
    assert isinstance(down, int) and down >= 1
    assert isinstance(groups, int) and groups >= 1
    (out_channels, in_channels_per_group, kh, kw) = _get_weight_shape(w)
    (fw, fh) = _get_filter_size(f)
    (px0, px1, py0, py1) = _parse_padding(padding)
    if up > 1:
        px0 += (fw + up - 1) // 2
        px1 += (fw - up) // 2
        py0 += (fh + up - 1) // 2
        py1 += (fh - up) // 2
    if down > 1:
        px0 += (fw - down + 1) // 2
        px1 += (fw - down) // 2
        py0 += (fh - down + 1) // 2
        py1 += (fh - down) // 2
    if kw == 1 and kh == 1 and (down > 1 and up == 1):
        x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        return x
    if kw == 1 and kh == 1 and (up > 1 and down == 1):
        x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
        x = upfirdn2d.upfirdn2d(x=x, f=f, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
        return x
    if down > 1 and up == 1:
        x = upfirdn2d.upfirdn2d(x=x, f=f, padding=[px0, px1, py0, py1], flip_filter=flip_filter)
        x = _conv2d_wrapper(x=x, w=w, stride=down, groups=groups, flip_weight=flip_weight)
        return x
    if up > 1:
        if groups == 1:
            w = w.transpose(0, 1)
        else:
            w = w.reshape(groups, out_channels // groups, in_channels_per_group, kh, kw)
            w = w.transpose(1, 2)
            w = w.reshape(groups * in_channels_per_group, out_channels // groups, kh, kw)
        px0 -= kw - 1
        px1 -= kw - up
        py0 -= kh - 1
        py1 -= kh - up
        pxt = max(min(-px0, -px1), 0)
        pyt = max(min(-py0, -py1), 0)
        x = _conv2d_wrapper(x=x, w=w, stride=up, padding=[pyt, pxt], groups=groups, transpose=True, flip_weight=not flip_weight)
        x = upfirdn2d.upfirdn2d(x=x, f=f, padding=[px0 + pxt, px1 + pxt, py0 + pyt, py1 + pyt], gain=up ** 2, flip_filter=flip_filter)
        if down > 1:
            x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
        return x
    if up == 1 and down == 1:
        if px0 == px1 and py0 == py1 and (px0 >= 0) and (py0 >= 0):
            return _conv2d_wrapper(x=x, w=w, padding=[py0, px0], groups=groups, flip_weight=flip_weight)
    x = upfirdn2d.upfirdn2d(x=x, f=f if up > 1 else None, up=up, padding=[px0, px1, py0, py1], gain=up ** 2, flip_filter=flip_filter)
    x = _conv2d_wrapper(x=x, w=w, groups=groups, flip_weight=flip_weight)
    if down > 1:
        x = upfirdn2d.upfirdn2d(x=x, f=f, down=down, flip_filter=flip_filter)
    return x