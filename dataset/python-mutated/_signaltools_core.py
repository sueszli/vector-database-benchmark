import math
import cupy
from cupy._core import internal
from cupyx.scipy import fft
from cupyx.scipy.ndimage import _filters
from cupyx.scipy.ndimage import _util

def _check_conv_inputs(in1, in2, mode, convolution=True):
    if False:
        print('Hello World!')
    if in1.ndim == in2.ndim == 0:
        return in1 * (in2 if convolution else in2.conj())
    if in1.ndim != in2.ndim:
        raise ValueError('in1 and in2 should have the same dimensionality')
    if in1.size == 0 or in2.size == 0:
        return cupy.array([], dtype=in1.dtype)
    if mode not in ('full', 'same', 'valid'):
        raise ValueError('acceptable modes are "valid", "same", or "full"')
    return None

def _direct_correlate(in1, in2, mode='full', output=float, convolution=False, boundary='constant', fillvalue=0.0, shift=False):
    if False:
        print('Hello World!')
    if in1.ndim != 1 and (in1.dtype.kind == 'b' or (in1.dtype.kind == 'f' and in1.dtype.itemsize < 4)):
        raise ValueError('unsupported type in SciPy')
    swapped_inputs = False
    orig_in1_shape = in1.shape
    if _inputs_swap_needed(mode, in1.shape, in2.shape) or (in2.size > in1.size and boundary == 'constant' and (fillvalue == 0)):
        (in1, in2) = (in2, in1)
        swapped_inputs = True
    if in2.nbytes >= 1 << 31:
        raise RuntimeError('smaller array must be 2 GiB or less, use method="fft" instead')
    if mode == 'full':
        out_shape = tuple((x1 + x2 - 1 for (x1, x2) in zip(in1.shape, in2.shape)))
        offsets = tuple((x - 1 for x in in2.shape))
    elif mode == 'valid':
        out_shape = tuple((x1 - x2 + 1 for (x1, x2) in zip(in1.shape, in2.shape)))
        offsets = (0,) * in1.ndim
    else:
        out_shape = orig_in1_shape
        if orig_in1_shape == in1.shape:
            offsets = tuple(((x - shift) // 2 for x in in2.shape))
        else:
            offsets = tuple(((2 * x2 - x1 - (not convolution) + shift) // 2 for (x1, x2) in zip(in1.shape, in2.shape)))
    out_dtype = cupy.promote_types(in1, in2)
    if not isinstance(output, cupy.ndarray):
        if not cupy.can_cast(output, out_dtype):
            raise ValueError('not available for this type')
        output = cupy.empty(out_shape, out_dtype)
    elif output.shape != out_shape:
        raise ValueError('out has wrong shape')
    elif output.dtype != out_dtype:
        raise ValueError('out has wrong dtype')
    if cupy.can_cast(in2, in1):
        in2 = in2.astype(out_dtype)
    int_type = _util._get_inttype(in1)
    kernel = _filters._get_correlate_kernel(boundary, in2.shape, int_type, offsets, fillvalue)
    in2 = _reverse(in2) if convolution else in2.conj()
    if not swapped_inputs or convolution:
        kernel(in1, in2, output)
    elif output.dtype.kind != 'c':
        kernel(in1, in2, _reverse(output))
    else:
        kernel(in1, in2, output)
        output = cupy.ascontiguousarray(_reverse(output))
        if swapped_inputs and (mode != 'valid' or not shift):
            cupy.conjugate(output, out=output)
    return output

def _reverse(x):
    if False:
        print('Hello World!')
    return x[(slice(None, None, -1),) * x.ndim]

def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    if False:
        print('Hello World!')
    if mode != 'valid' or not shape1:
        return False
    if axes is None:
        axes = tuple(range(len(shape1)))
    not_ok1 = any((shape1[i] < shape2[i] for i in axes))
    not_ok2 = any((shape1[i] > shape2[i] for i in axes))
    if not_ok1 and not_ok2:
        raise ValueError('For "valid" mode, one must be at least as large as the other in every dimension')
    return not_ok1

def _init_freq_conv_axes(in1, in2, mode, axes, sorted_axes=False):
    if False:
        for i in range(10):
            print('nop')
    (s1, s2) = (in1.shape, in2.shape)
    axes = _init_nd_and_axes(in1, axes)
    axes = [ax for ax in axes if s1[ax] != 1 and s2[ax] != 1]
    if sorted_axes:
        axes.sort()
    for (ax, (dim1, dim2)) in enumerate(zip(s1, s2)):
        if ax not in axes and dim1 != dim2 and (dim1 != 1) and (dim2 != 1):
            raise ValueError('incompatible shapes for in1 and in2: {} and {}'.format(s1, s2))
    if _inputs_swap_needed(mode, s1, s2, axes=axes):
        (in1, in2) = (in2, in1)
    return (in1, in2, tuple(axes))

def _init_nd_and_axes(x, axes):
    if False:
        print('Hello World!')
    axes = internal._normalize_axis_indices(axes, x.ndim, sort_axes=False)
    if not len(axes):
        raise ValueError('when provided, axes cannot be empty')
    if any((x.shape[ax] < 1 for ax in axes)):
        raise ValueError('invalid number of data points specified')
    return axes

def _freq_domain_conv(in1, in2, axes, shape, calc_fast_len=False):
    if False:
        print('Hello World!')
    if not axes:
        return in1 * in2
    real = in1.dtype.kind != 'c' and in2.dtype.kind != 'c'
    fshape = [fft.next_fast_len(shape[a], real) for a in axes] if calc_fast_len else shape
    (fftn, ifftn) = (fft.rfftn, fft.irfftn) if real else (fft.fftn, fft.ifftn)
    sp1 = fftn(in1, fshape, axes=axes)
    sp2 = fftn(in2, fshape, axes=axes)
    out = ifftn(sp1 * sp2, fshape, axes=axes)
    return out[tuple((slice(x) for x in shape))] if calc_fast_len else out

def _apply_conv_mode(full, s1, s2, mode, axes):
    if False:
        print('Hello World!')
    if mode == 'full':
        return cupy.ascontiguousarray(full)
    if mode == 'valid':
        s1 = [full.shape[a] if a not in axes else s1[a] - s2[a] + 1 for a in range(full.ndim)]
    starts = [(cur - new) // 2 for (cur, new) in zip(full.shape, s1)]
    slices = tuple((slice(start, start + length) for (start, length) in zip(starts, s1)))
    return cupy.ascontiguousarray(full[slices])
__EXP_N1 = 0.36787944117144233

def _optimal_oa_block_size(overlap):
    if False:
        for i in range(10):
            print('nop')
    '\n    Computes the optimal block size for the OA method given the overlap size.\n\n    Computed as ``ceil(-overlap*W(-1/(2*e*overlap)))`` where ``W(z)`` is the\n    Lambert W function solved as per ``scipy.special.lambertw(z, -1)`` with a\n    fixed 4 iterations.\n\n    Returned size should still be given to ``cupyx.scipy.fft.next_fast_len()``.\n    '
    z = -__EXP_N1 / (2 * overlap)
    w = -1 - math.log(2 * overlap)
    for i in range(4):
        ew = math.exp(w)
        wew = w * ew
        wewz = wew - z
        w -= wewz / (wew + ew - (w + 2) * wewz / (2 * w + 2))
    return math.ceil(-overlap * w)

def _calc_oa_lens(s1, s2):
    if False:
        for i in range(10):
            print('nop')
    fallback = (s1 + s2 - 1, None, s1, s2)
    if s1 == s2 or s1 == 1 or s2 == 1:
        return fallback
    swapped = s2 > s1
    if swapped:
        (s1, s2) = (s2, s1)
    if s2 >= s1 // 2:
        return fallback
    overlap = s2 - 1
    block_size = fft.next_fast_len(_optimal_oa_block_size(overlap))
    if block_size >= s1:
        return fallback
    (in1_step, in2_step) = (block_size - s2 + 1, s2)
    if swapped:
        (in1_step, in2_step) = (in2_step, in1_step)
    return (block_size, overlap, in1_step, in2_step)

def _oa_reshape_inputs(in1, in2, axes, shape_final, block_size, overlaps, in1_step, in2_step):
    if False:
        for i in range(10):
            print('nop')
    nsteps1 = []
    nsteps2 = []
    pad_size1 = []
    pad_size2 = []
    for i in range(in1.ndim):
        if i not in axes:
            pad_size1 += [(0, 0)]
            pad_size2 += [(0, 0)]
            continue
        (curnstep1, curpad1, curnstep2, curpad2) = (1, 0, 1, 0)
        if in1.shape[i] > in1_step[i]:
            curnstep1 = math.ceil((in1.shape[i] + 1) / in1_step[i])
            if (block_size[i] - overlaps[i]) * curnstep1 < shape_final[i]:
                curnstep1 += 1
            curpad1 = curnstep1 * in1_step[i] - in1.shape[i]
        if in2.shape[i] > in2_step[i]:
            curnstep2 = math.ceil((in2.shape[i] + 1) / in2_step[i])
            if (block_size[i] - overlaps[i]) * curnstep2 < shape_final[i]:
                curnstep2 += 1
            curpad2 = curnstep2 * in2_step[i] - in2.shape[i]
        nsteps1 += [curnstep1]
        nsteps2 += [curnstep2]
        pad_size1 += [(0, curpad1)]
        pad_size2 += [(0, curpad2)]
    if not all((curpad == (0, 0) for curpad in pad_size1)):
        in1 = cupy.pad(in1, pad_size1, mode='constant', constant_values=0)
    if not all((curpad == (0, 0) for curpad in pad_size2)):
        in2 = cupy.pad(in2, pad_size2, mode='constant', constant_values=0)
    reshape_size1 = list(in1_step)
    reshape_size2 = list(in2_step)
    for (i, iax) in enumerate(axes):
        reshape_size1.insert(iax + i, nsteps1[i])
        reshape_size2.insert(iax + i, nsteps2[i])
    return (in1.reshape(*reshape_size1), in2.reshape(*reshape_size2))