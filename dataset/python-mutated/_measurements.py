import warnings
import numpy
import cupy
from cupy import _core
from cupy import _util

def label(input, structure=None, output=None):
    if False:
        print('Hello World!')
    'Labels features in an array.\n\n    Args:\n        input (cupy.ndarray): The input array.\n        structure (array_like or None): A structuring element that defines\n            feature connections. ```structure``` must be centersymmetric. If\n            None, structure is automatically generated with a squared\n            connectivity equal to one.\n        output (cupy.ndarray, dtype or None): The array in which to place the\n            output.\n    Returns:\n        label (cupy.ndarray): An integer array where each unique feature in\n        ```input``` has a unique label in the array.\n\n        num_features (int): Number of features found.\n\n    .. warning::\n\n        This function may synchronize the device.\n\n    .. seealso:: :func:`scipy.ndimage.label`\n    '
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype.char in 'FD':
        raise TypeError('Complex type not supported')
    if structure is None:
        structure = _generate_binary_structure(input.ndim, 1)
    elif isinstance(structure, cupy.ndarray):
        structure = cupy.asnumpy(structure)
    structure = numpy.array(structure, dtype=bool)
    if structure.ndim != input.ndim:
        raise RuntimeError('structure and input must have equal rank')
    for i in structure.shape:
        if i != 3:
            raise ValueError('structure dimensions must be equal to 3')
    if isinstance(output, cupy.ndarray):
        if output.shape != input.shape:
            raise ValueError('output shape not correct')
        caller_provided_output = True
    else:
        caller_provided_output = False
        if output is None:
            output = cupy.empty(input.shape, numpy.int32)
        else:
            output = cupy.empty(input.shape, output)
    if input.size == 0:
        maxlabel = 0
    elif input.ndim == 0:
        maxlabel = 0 if input.item() == 0 else 1
        output.fill(maxlabel)
    else:
        if output.dtype != numpy.int32:
            y = cupy.empty(input.shape, numpy.int32)
        else:
            y = output
        maxlabel = _label(input, structure, y)
        if output.dtype != numpy.int32:
            _core.elementwise_copy(y, output)
    if caller_provided_output:
        return maxlabel
    else:
        return (output, maxlabel)

def _generate_binary_structure(rank, connectivity):
    if False:
        for i in range(10):
            print('nop')
    if connectivity < 1:
        connectivity = 1
    if rank < 1:
        return numpy.array(True, dtype=bool)
    output = numpy.fabs(numpy.indices([3] * rank) - 1)
    output = numpy.add.reduce(output, 0)
    return output <= connectivity

def _label(x, structure, y):
    if False:
        i = 10
        return i + 15
    elems = numpy.where(structure != 0)
    vecs = [elems[dm] - 1 for dm in range(x.ndim)]
    offset = vecs[0]
    for dm in range(1, x.ndim):
        offset = offset * 3 + vecs[dm]
    indxs = numpy.where(offset < 0)[0]
    dirs = [[vecs[dm][dr] for dm in range(x.ndim)] for dr in indxs]
    dirs = cupy.array(dirs, dtype=numpy.int32)
    ndirs = indxs.shape[0]
    y_shape = cupy.array(y.shape, dtype=numpy.int32)
    count = cupy.zeros(2, dtype=numpy.int32)
    _kernel_init()(x, y)
    _kernel_connect()(y_shape, dirs, ndirs, x.ndim, y, size=y.size)
    _kernel_count()(y, count, size=y.size)
    maxlabel = int(count[0])
    labels = cupy.empty(maxlabel, dtype=numpy.int32)
    _kernel_labels()(y, count, labels, size=y.size)
    _kernel_finalize()(maxlabel, cupy.sort(labels), y, size=y.size)
    return maxlabel

def _kernel_init():
    if False:
        while True:
            i = 10
    return _core.ElementwiseKernel('X x', 'Y y', 'if (x == 0) { y = -1; } else { y = i; }', 'cupyx_scipy_ndimage_label_init')

def _kernel_connect():
    if False:
        while True:
            i = 10
    return _core.ElementwiseKernel('raw int32 shape, raw int32 dirs, int32 ndirs, int32 ndim', 'raw Y y', '\n        if (y[i] < 0) continue;\n        for (int dr = 0; dr < ndirs; dr++) {\n            int j = i;\n            int rest = j;\n            int stride = 1;\n            int k = 0;\n            for (int dm = ndim-1; dm >= 0; dm--) {\n                int pos = rest % shape[dm] + dirs[dm + dr * ndim];\n                if (pos < 0 || pos >= shape[dm]) {\n                    k = -1;\n                    break;\n                }\n                k += pos * stride;\n                rest /= shape[dm];\n                stride *= shape[dm];\n            }\n            if (k < 0) continue;\n            if (y[k] < 0) continue;\n            while (1) {\n                while (j != y[j]) { j = y[j]; }\n                while (k != y[k]) { k = y[k]; }\n                if (j == k) break;\n                if (j < k) {\n                    int old = atomicCAS( &y[k], k, j );\n                    if (old == k) break;\n                    k = old;\n                }\n                else {\n                    int old = atomicCAS( &y[j], j, k );\n                    if (old == j) break;\n                    j = old;\n                }\n            }\n        }\n        ', 'cupyx_scipy_ndimage_label_connect')

def _kernel_count():
    if False:
        for i in range(10):
            print('nop')
    return _core.ElementwiseKernel('', 'raw Y y, raw int32 count', '\n        if (y[i] < 0) continue;\n        int j = i;\n        while (j != y[j]) { j = y[j]; }\n        if (j != i) y[i] = j;\n        else atomicAdd(&count[0], 1);\n        ', 'cupyx_scipy_ndimage_label_count')

def _kernel_labels():
    if False:
        i = 10
        return i + 15
    return _core.ElementwiseKernel('', 'raw Y y, raw int32 count, raw int32 labels', '\n        if (y[i] != i) continue;\n        int j = atomicAdd(&count[1], 1);\n        labels[j] = i;\n        ', 'cupyx_scipy_ndimage_label_labels')

def _kernel_finalize():
    if False:
        print('Hello World!')
    return _core.ElementwiseKernel('int32 maxlabel', 'raw int32 labels, raw Y y', '\n        if (y[i] < 0) {\n            y[i] = 0;\n            continue;\n        }\n        int yi = y[i];\n        int j_min = 0;\n        int j_max = maxlabel - 1;\n        int j = (j_min + j_max) / 2;\n        while (j_min < j_max) {\n            if (yi == labels[j]) break;\n            if (yi < labels[j]) j_max = j - 1;\n            else j_min = j + 1;\n            j = (j_min + j_max) / 2;\n        }\n        y[i] = j + 1;\n        ', 'cupyx_scipy_ndimage_label_finalize')
_ndimage_variance_kernel = _core.ElementwiseKernel('T input, R labels, raw X index, uint64 size, raw float64 mean', 'raw float64 out', '\n    for (ptrdiff_t j = 0; j < size; j++) {\n      if (labels == index[j]) {\n        atomicAdd(&out[j], (input - mean[j]) * (input - mean[j]));\n        break;\n      }\n    }\n    ', 'cupyx_scipy_ndimage_variance')
_ndimage_sum_kernel = _core.ElementwiseKernel('T input, R labels, raw X index, uint64 size', 'raw float64 out', '\n    for (ptrdiff_t j = 0; j < size; j++) {\n      if (labels == index[j]) {\n        atomicAdd(&out[j], input);\n        break;\n      }\n    }\n    ', 'cupyx_scipy_ndimage_sum')

def _ndimage_sum_kernel_2(input, labels, index, sum_val, batch_size=4):
    if False:
        print('Hello World!')
    for i in range(0, index.size, batch_size):
        matched = labels == index[i:i + batch_size].reshape((-1,) + (1,) * input.ndim)
        sum_axes = tuple(range(1, 1 + input.ndim))
        sum_val[i:i + batch_size] = cupy.where(matched, input, 0).sum(axis=sum_axes)
    return sum_val
_ndimage_mean_kernel = _core.ElementwiseKernel('T input, R labels, raw X index, uint64 size', 'raw float64 out, raw uint64 count', '\n    for (ptrdiff_t j = 0; j < size; j++) {\n      if (labels == index[j]) {\n        atomicAdd(&out[j], input);\n        atomicAdd(&count[j], 1);\n        break;\n      }\n    }\n    ', 'cupyx_scipy_ndimage_mean')

def _ndimage_mean_kernel_2(input, labels, index, batch_size=4, return_count=False):
    if False:
        i = 10
        return i + 15
    sum_val = cupy.empty_like(index, dtype=cupy.float64)
    count = cupy.empty_like(index, dtype=cupy.uint64)
    for i in range(0, index.size, batch_size):
        matched = labels == index[i:i + batch_size].reshape((-1,) + (1,) * input.ndim)
        mean_axes = tuple(range(1, 1 + input.ndim))
        count[i:i + batch_size] = matched.sum(axis=mean_axes)
        sum_val[i:i + batch_size] = cupy.where(matched, input, 0).sum(axis=mean_axes)
    if return_count:
        return (sum_val / count, count)
    return sum_val / count

def _mean_driver(input, labels, index, return_count=False, use_kern=False):
    if False:
        while True:
            i = 10
    if use_kern:
        return _ndimage_mean_kernel_2(input, labels, index, return_count=return_count)
    out = cupy.zeros_like(index, cupy.float64)
    count = cupy.zeros_like(index, dtype=cupy.uint64)
    (sum, count) = _ndimage_mean_kernel(input, labels, index, index.size, out, count)
    if return_count:
        return (sum / count, count)
    return sum / count

def variance(input, labels=None, index=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculates the variance of the values of an n-D image array, optionally\n    at specified sub-regions.\n\n    Args:\n        input (cupy.ndarray): Nd-image data to process.\n        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.\n            If not None, must be same shape as `input`.\n        index (cupy.ndarray or None): `labels` to include in output. If None\n            (default), all values where `labels` is non-zero are used.\n\n    Returns:\n        cupy.ndarray: Values of variance, for each sub-region if\n        `labels` and `index` are specified.\n\n    .. seealso:: :func:`scipy.ndimage.variance`\n    '
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype in (cupy.complex64, cupy.complex128):
        raise TypeError("cupyx.scipy.ndimage.variance doesn't support %{}".format(input.dtype.type))
    use_kern = False
    if input.dtype not in [cupy.int32, cupy.float16, cupy.float32, cupy.float64, cupy.uint32, cupy.uint64, cupy.ulonglong]:
        warnings.warn(f'Using the slower implementation because the provided type {input.dtype} is not supported by cupyx.scipy.ndimage.sum. Consider using an array of type int32, float16, float32, float64, uint32, uint64 as data types for the fast implementation', _util.PerformanceWarning)
        use_kern = True

    def calc_var_with_intermediate_float(input):
        if False:
            while True:
                i = 10
        vals_c = input - input.mean()
        count = vals_c.size
        return cupy.square(vals_c).sum() / cupy.asanyarray(count).astype(float)
    if labels is None:
        return calc_var_with_intermediate_float(input)
    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')
    (input, labels) = cupy.broadcast_arrays(input, labels)
    if index is None:
        return calc_var_with_intermediate_float(input[labels > 0])
    if cupy.isscalar(index):
        return calc_var_with_intermediate_float(input[labels == index])
    if not isinstance(index, cupy.ndarray):
        if not isinstance(index, int):
            raise TypeError('index must be cupy.ndarray or a scalar int')
        else:
            return input[labels == index].var().astype(cupy.float64, copy=False)
    (mean_val, count) = _mean_driver(input, labels, index, True, use_kern)
    if use_kern:
        new_axis = (..., *(cupy.newaxis for _ in range(input.ndim)))
        return cupy.where(labels[None, ...] == index[new_axis], cupy.square(input - mean_val[new_axis]), 0).sum(tuple(range(1, input.ndim + 1))) / count
    out = cupy.zeros_like(index, dtype=cupy.float64)
    return _ndimage_variance_kernel(input, labels, index, index.size, mean_val, out) / count

def sum_labels(input, labels=None, index=None):
    if False:
        while True:
            i = 10
    'Calculates the sum of the values of an n-D image array, optionally\n       at specified sub-regions.\n\n    Args:\n        input (cupy.ndarray): Nd-image data to process.\n        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.\n            If not None, must be same shape as `input`.\n        index (cupy.ndarray or None): `labels` to include in output. If None\n            (default), all values where `labels` is non-zero are used.\n\n    Returns:\n       sum (cupy.ndarray): sum of values, for each sub-region if\n       `labels` and `index` are specified.\n\n    .. seealso:: :func:`scipy.ndimage.sum_labels`\n    '
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype in (cupy.complex64, cupy.complex128):
        raise TypeError('cupyx.scipy.ndimage.sum does not support %{}'.format(input.dtype.type))
    use_kern = False
    if input.dtype not in [cupy.int32, cupy.float16, cupy.float32, cupy.float64, cupy.uint32, cupy.uint64, cupy.ulonglong]:
        warnings.warn('Using the slower implmentation as cupyx.scipy.ndimage.sum supports int32, float16, float32, float64, uint32, uint64 as data typesfor the fast implmentation', _util.PerformanceWarning)
        use_kern = True
    if labels is None:
        return input.sum()
    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')
    (input, labels) = cupy.broadcast_arrays(input, labels)
    if index is None:
        return input[labels != 0].sum()
    if not isinstance(index, cupy.ndarray):
        if not isinstance(index, int):
            raise TypeError('index must be cupy.ndarray or a scalar int')
        else:
            return input[labels == index].sum()
    if index.size == 0:
        return cupy.array([], dtype=cupy.int64)
    out = cupy.zeros_like(index, dtype=cupy.float64)
    if input.size >= 262144 and index.size <= 4 or use_kern:
        return _ndimage_sum_kernel_2(input, labels, index, out)
    return _ndimage_sum_kernel(input, labels, index, index.size, out)

def sum(input, labels=None, index=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculates the sum of the values of an n-D image array, optionally\n       at specified sub-regions.\n\n    Args:\n        input (cupy.ndarray): Nd-image data to process.\n        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.\n            If not None, must be same shape as `input`.\n        index (cupy.ndarray or None): `labels` to include in output. If None\n            (default), all values where `labels` is non-zero are used.\n\n    Returns:\n       sum (cupy.ndarray): sum of values, for each sub-region if\n       `labels` and `index` are specified.\n\n    Notes:\n        This is an alias for `cupyx.scipy.ndimage.sum_labels` kept for\n        backwards compatibility reasons. For new code please prefer\n        `sum_labels`.\n\n    .. seealso:: :func:`scipy.ndimage.sum`\n    '
    return sum_labels(input, labels, index)

def mean(input, labels=None, index=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculates the mean of the values of an n-D image array, optionally\n       at specified sub-regions.\n\n    Args:\n        input (cupy.ndarray): Nd-image data to process.\n        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.\n            If not None, must be same shape as `input`.\n        index (cupy.ndarray or None): `labels` to include in output. If None\n            (default), all values where `labels` is non-zero are used.\n\n    Returns:\n        mean (cupy.ndarray): mean of values, for each sub-region if\n        `labels` and `index` are specified.\n\n\n    .. seealso:: :func:`scipy.ndimage.mean`\n    '
    if not isinstance(input, cupy.ndarray):
        raise TypeError('input must be cupy.ndarray')
    if input.dtype in (cupy.complex64, cupy.complex128):
        raise TypeError('cupyx.scipy.ndimage.mean does not support %{}'.format(input.dtype.type))
    use_kern = False
    if input.dtype not in [cupy.int32, cupy.float16, cupy.float32, cupy.float64, cupy.uint32, cupy.uint64, cupy.ulonglong]:
        warnings.warn('Using the slower implmentation as cupyx.scipy.ndimage.mean supports int32, float16, float32, float64, uint32, uint64 as data types for the fast implmentation', _util.PerformanceWarning)
        use_kern = True

    def calc_mean_with_intermediate_float(input):
        if False:
            while True:
                i = 10
        sum = input.sum()
        count = input.size
        return sum / cupy.asanyarray(count).astype(float)
    if labels is None:
        return calc_mean_with_intermediate_float(input)
    if not isinstance(labels, cupy.ndarray):
        raise TypeError('label must be cupy.ndarray')
    (input, labels) = cupy.broadcast_arrays(input, labels)
    if index is None:
        return calc_mean_with_intermediate_float(input[labels > 0])
    if cupy.isscalar(index):
        return calc_mean_with_intermediate_float(input[labels == index])
    if not isinstance(index, cupy.ndarray):
        if not isinstance(index, int):
            raise TypeError('index must be cupy.ndarray or a scalar int')
        else:
            return input[labels == index].mean(dtype=cupy.float64)
    return _mean_driver(input, labels, index, use_kern=use_kern)

def standard_deviation(input, labels=None, index=None):
    if False:
        return 10
    'Calculates the standard deviation of the values of an n-D image array,\n    optionally at specified sub-regions.\n\n    Args:\n        input (cupy.ndarray): Nd-image data to process.\n        labels (cupy.ndarray or None): Labels defining sub-regions in `input`.\n            If not None, must be same shape as `input`.\n        index (cupy.ndarray or None): `labels` to include in output. If None\n            (default), all values where `labels` is non-zero are used.\n\n    Returns:\n        standard_deviation (cupy.ndarray): standard deviation of values, for\n        each sub-region if `labels` and `index` are specified.\n\n    .. seealso:: :func:`scipy.ndimage.standard_deviation`\n    '
    return cupy.sqrt(variance(input, labels, index))

def _safely_castable_to_int(dt):
    if False:
        while True:
            i = 10
    'Test whether the NumPy data type `dt` can be safely cast to an int.'
    int_size = cupy.dtype(int).itemsize
    safe = cupy.issubdtype(dt, cupy.signedinteger) and dt.itemsize <= int_size or (cupy.issubdtype(dt, cupy.unsignedinteger) and dt.itemsize < int_size)
    return safe

def _get_values(arrays, func):
    if False:
        while True:
            i = 10
    'Concatenated result of applying func to a list of arrays.\n\n    func should be cupy.min, cupy.max or cupy.median\n    '
    dtype = arrays[0].dtype
    return cupy.concatenate([func(a, keepdims=True) if a.size != 0 else cupy.asarray([0], dtype=dtype) for a in arrays])

def _get_positions(arrays, position_arrays, arg_func):
    if False:
        i = 10
        return i + 15
    'Concatenated positions from applying arg_func to arrays.\n\n    arg_func should be cupy.argmin or cupy.argmax\n    '
    return cupy.concatenate([pos[arg_func(a, keepdims=True)] if a.size != 0 else cupy.asarray([0], dtype=int) for (pos, a) in zip(position_arrays, arrays)])

def _select_via_looping(input, labels, idxs, positions, find_min, find_min_positions, find_max, find_max_positions, find_median):
    if False:
        i = 10
        return i + 15
    'Internal helper routine for _select.\n\n    With relatively few labels it is faster to call this function rather than\n    using the implementation based on cupy.lexsort.\n    '
    find_positions = find_min_positions or find_max_positions
    arrays = []
    position_arrays = []
    for i in idxs:
        label_idx = labels == i
        arrays.append(input[label_idx])
        if find_positions:
            position_arrays.append(positions[label_idx])
    result = []
    if find_min:
        result += [_get_values(arrays, cupy.min)]
    if find_min_positions:
        result += [_get_positions(arrays, position_arrays, cupy.argmin)]
    if find_max:
        result += [_get_values(arrays, cupy.max)]
    if find_max_positions:
        result += [_get_positions(arrays, position_arrays, cupy.argmax)]
    if find_median:
        result += [_get_values(arrays, cupy.median)]
    return result

def _select(input, labels=None, index=None, find_min=False, find_max=False, find_min_positions=False, find_max_positions=False, find_median=False):
    if False:
        i = 10
        return i + 15
    'Return one or more of: min, max, min position, max position, median.\n\n    If neither `labels` or `index` is provided, these are the global values\n    in `input`. If `index` is None, but `labels` is provided, a global value\n    across all non-zero labels is given. When both `labels` and `index` are\n    provided, lists of values are provided for each labeled region specified\n    in `index`. See further details in :func:`cupyx.scipy.ndimage.minimum`,\n    etc.\n\n    Used by minimum, maximum, minimum_position, maximum_position, extrema.\n    '
    find_positions = find_min_positions or find_max_positions
    positions = None
    if find_positions:
        positions = cupy.arange(input.size).reshape(input.shape)

    def single_group(vals, positions):
        if False:
            print('Hello World!')
        result = []
        if find_min:
            result += [vals.min()]
        if find_min_positions:
            result += [positions[vals == vals.min()][0]]
        if find_max:
            result += [vals.max()]
        if find_max_positions:
            result += [positions[vals == vals.max()][0]]
        if find_median:
            result += [cupy.median(vals)]
        return result
    if labels is None:
        return single_group(input, positions)
    (input, labels) = cupy.broadcast_arrays(input, labels)
    if index is None:
        mask = labels > 0
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)
    if cupy.isscalar(index):
        mask = labels == index
        masked_positions = None
        if find_positions:
            masked_positions = positions[mask]
        return single_group(input[mask], masked_positions)
    index = cupy.asarray(index)
    safe_int = _safely_castable_to_int(labels.dtype)
    min_label = labels.min()
    max_label = labels.max()
    if not safe_int or min_label < 0 or max_label > labels.size:
        (unique_labels, labels) = cupy.unique(labels, return_inverse=True)
        idxs = cupy.searchsorted(unique_labels, index)
        idxs[idxs >= unique_labels.size] = 0
        found = unique_labels[idxs] == index
    else:
        idxs = cupy.asanyarray(index, int).copy()
        found = (idxs >= 0) & (idxs <= max_label)
    idxs[~found] = max_label + 1
    input = input.ravel()
    labels = labels.ravel()
    if find_positions:
        positions = positions.ravel()
    using_cub = _core._accelerator.ACCELERATOR_CUB in cupy._core.get_routine_accelerators()
    if using_cub:
        if find_positions or find_median:
            n_label_cutoff = 15
        else:
            n_label_cutoff = 30
    else:
        n_label_cutoff = 0
    if n_label_cutoff and len(idxs) <= n_label_cutoff:
        return _select_via_looping(input, labels, idxs, positions, find_min, find_min_positions, find_max, find_max_positions, find_median)
    order = cupy.lexsort(cupy.stack((input.ravel(), labels.ravel())))
    input = input[order]
    labels = labels[order]
    if find_positions:
        positions = positions[order]
    label_change_index = cupy.searchsorted(labels, cupy.arange(1, max_label + 2))
    if find_min or find_min_positions or find_median:
        min_index = label_change_index[:-1]
    if find_max or find_max_positions or find_median:
        max_index = label_change_index[1:] - 1
    result = []
    if find_min:
        mins = cupy.zeros(int(labels.max()) + 2, input.dtype)
        mins[labels[min_index]] = input[min_index]
        result += [mins[idxs]]
    if find_min_positions:
        minpos = cupy.zeros(labels.max().item() + 2, int)
        minpos[labels[min_index]] = positions[min_index]
        result += [minpos[idxs]]
    if find_max:
        maxs = cupy.zeros(int(labels.max()) + 2, input.dtype)
        maxs[labels[max_index]] = input[max_index]
        result += [maxs[idxs]]
    if find_max_positions:
        maxpos = cupy.zeros(labels.max().item() + 2, int)
        maxpos[labels[max_index]] = positions[max_index]
        result += [maxpos[idxs]]
    if find_median:
        locs = cupy.arange(len(labels))
        lo = cupy.zeros(int(labels.max()) + 2, int)
        lo[labels[min_index]] = locs[min_index]
        hi = cupy.zeros(int(labels.max()) + 2, int)
        hi[labels[max_index]] = locs[max_index]
        lo = lo[idxs]
        hi = hi[idxs]
        step = (hi - lo) // 2
        lo += step
        hi -= step
        if input.dtype.kind in 'iub':
            result += [(input[lo].astype(float) + input[hi].astype(float)) / 2.0]
        else:
            result += [(input[lo] + input[hi]) / 2.0]
    return result

def minimum(input, labels=None, index=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculate the minimum of the values of an array over labeled regions.\n\n    Args:\n        input (cupy.ndarray):\n            Array of values. For each region specified by `labels`, the\n            minimal values of `input` over the region is computed.\n        labels (cupy.ndarray, optional): An array of integers marking different\n            regions over which the minimum value of `input` is to be computed.\n            `labels` must have the same shape as `input`. If `labels` is not\n            specified, the minimum over the whole array is returned.\n        index (array_like, optional): A list of region labels that are taken\n            into account for computing the minima. If `index` is None, the\n            minimum over all elements where `labels` is non-zero is returned.\n\n    Returns:\n        cupy.ndarray: Array of minima of `input` over the regions\n        determined by `labels` and whose index is in `index`. If `index` or\n        `labels` are not specified, a 0-dimensional cupy.ndarray is\n        returned: the minimal value of `input` if `labels` is None,\n        and the minimal value of elements where `labels` is greater than\n        zero if `index` is None.\n\n    .. seealso:: :func:`scipy.ndimage.minimum`\n    '
    return _select(input, labels, index, find_min=True)[0]

def maximum(input, labels=None, index=None):
    if False:
        while True:
            i = 10
    'Calculate the maximum of the values of an array over labeled regions.\n\n    Args:\n        input (cupy.ndarray):\n            Array of values. For each region specified by `labels`, the\n            maximal values of `input` over the region is computed.\n        labels (cupy.ndarray, optional): An array of integers marking different\n            regions over which the maximum value of `input` is to be computed.\n            `labels` must have the same shape as `input`. If `labels` is not\n            specified, the maximum over the whole array is returned.\n        index (array_like, optional): A list of region labels that are taken\n            into account for computing the maxima. If `index` is None, the\n            maximum over all elements where `labels` is non-zero is returned.\n\n    Returns:\n        cupy.ndarray: Array of maxima of `input` over the regions\n        determaxed by `labels` and whose index is in `index`. If `index` or\n        `labels` are not specified, a 0-dimensional cupy.ndarray is\n        returned: the maximal value of `input` if `labels` is None,\n        and the maximal value of elements where `labels` is greater than\n        zero if `index` is None.\n\n    .. seealso:: :func:`scipy.ndimage.maximum`\n    '
    return _select(input, labels, index, find_max=True)[0]

def median(input, labels=None, index=None):
    if False:
        for i in range(10):
            print('nop')
    'Calculate the median of the values of an array over labeled regions.\n\n    Args:\n        input (cupy.ndarray):\n            Array of values. For each region specified by `labels`, the\n            median values of `input` over the region is computed.\n        labels (cupy.ndarray, optional): An array of integers marking different\n            regions over which the median value of `input` is to be computed.\n            `labels` must have the same shape as `input`. If `labels` is not\n            specified, the median over the whole array is returned.\n        index (array_like, optional): A list of region labels that are taken\n            into account for computing the medians. If `index` is None, the\n            median over all elements where `labels` is non-zero is returned.\n\n    Returns:\n        cupy.ndarray: Array of medians of `input` over the regions\n        determined by `labels` and whose index is in `index`. If `index` or\n        `labels` are not specified, a 0-dimensional cupy.ndarray is\n        returned: the median value of `input` if `labels` is None,\n        and the median value of elements where `labels` is greater than\n        zero if `index` is None.\n\n    .. seealso:: :func:`scipy.ndimage.median`\n    '
    return _select(input, labels, index, find_median=True)[0]

def minimum_position(input, labels=None, index=None):
    if False:
        print('Hello World!')
    'Find the positions of the minimums of the values of an array at labels.\n\n    For each region specified by `labels`, the position of the minimum\n    value of `input` within the region is returned.\n\n    Args:\n        input (cupy.ndarray):\n            Array of values. For each region specified by `labels`, the\n            minimal values of `input` over the region is computed.\n        labels (cupy.ndarray, optional): An array of integers marking different\n            regions over which the position of the minimum value of `input` is\n            to be computed. `labels` must have the same shape as `input`. If\n            `labels` is not specified, the location of the first minimum over\n            the whole array is returned.\n\n            The `labels` argument only works when `index` is specified.\n        index (array_like, optional): A list of region labels that are taken\n            into account for finding the location of the minima. If `index` is\n            None, the ``first`` minimum over all elements where `labels` is\n            non-zero is returned.\n\n            The `index` argument only works when `labels` is specified.\n\n    Returns:\n        Tuple of ints or list of tuples of ints that specify the location of\n        minima of `input` over the regions determined by `labels` and  whose\n        index is in `index`.\n\n        If `index` or `labels` are not specified, a tuple of ints is returned\n        specifying the location of the first minimal value of `input`.\n\n    .. note::\n        When `input` has multiple identical minima within a labeled region,\n        the coordinates returned are not guaranteed to match those returned by\n        SciPy.\n\n    .. seealso:: :func:`scipy.ndimage.minimum_position`\n    '
    dims = numpy.asarray(input.shape)
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
    result = _select(input, labels, index, find_min_positions=True)[0]
    if result.ndim == 0:
        result = int(result)
    else:
        result = cupy.asnumpy(result)
    if cupy.isscalar(result):
        return tuple(result // dim_prod % dims)
    return [tuple(v) for v in result.reshape(-1, 1) // dim_prod % dims]

def maximum_position(input, labels=None, index=None):
    if False:
        for i in range(10):
            print('nop')
    'Find the positions of the maximums of the values of an array at labels.\n\n    For each region specified by `labels`, the position of the maximum\n    value of `input` within the region is returned.\n\n    Args:\n        input (cupy.ndarray):\n            Array of values. For each region specified by `labels`, the\n            maximal values of `input` over the region is computed.\n        labels (cupy.ndarray, optional): An array of integers marking different\n            regions over which the position of the maximum value of `input` is\n            to be computed. `labels` must have the same shape as `input`. If\n            `labels` is not specified, the location of the first maximum over\n            the whole array is returned.\n\n            The `labels` argument only works when `index` is specified.\n        index (array_like, optional): A list of region labels that are taken\n            into account for finding the location of the maxima. If `index` is\n            None, the ``first`` maximum over all elements where `labels` is\n            non-zero is returned.\n\n            The `index` argument only works when `labels` is specified.\n\n    Returns:\n        Tuple of ints or list of tuples of ints that specify the location of\n        maxima of `input` over the regions determaxed by `labels` and  whose\n        index is in `index`.\n\n        If `index` or `labels` are not specified, a tuple of ints is returned\n        specifying the location of the first maximal value of `input`.\n\n    .. note::\n        When `input` has multiple identical maxima within a labeled region,\n        the coordinates returned are not guaranteed to match those returned by\n        SciPy.\n\n    .. seealso:: :func:`scipy.ndimage.maximum_position`\n    '
    dims = numpy.asarray(input.shape)
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
    result = _select(input, labels, index, find_max_positions=True)[0]
    if result.ndim == 0:
        result = int(result)
    else:
        result = cupy.asnumpy(result)
    if cupy.isscalar(result):
        return tuple(result // dim_prod % dims)
    return [tuple(v) for v in result.reshape(-1, 1) // dim_prod % dims]

def extrema(input, labels=None, index=None):
    if False:
        while True:
            i = 10
    'Calculate the minimums and maximums of the values of an array at labels,\n    along with their positions.\n\n    Args:\n        input (cupy.ndarray): N-D image data to process.\n        labels (cupy.ndarray, optional): Labels of features in input. If not\n            None, must be same shape as `input`.\n        index (int or sequence of ints, optional): Labels to include in output.\n            If None (default), all values where non-zero `labels` are used.\n\n    Returns:\n        A tuple that contains the following values.\n\n        **minimums (cupy.ndarray)**: Values of minimums in each feature.\n\n        **maximums (cupy.ndarray)**: Values of maximums in each feature.\n\n        **min_positions (tuple or list of tuples)**: Each tuple gives the N-D\n        coordinates of the corresponding minimum.\n\n        **max_positions (tuple or list of tuples)**: Each tuple gives the N-D\n        coordinates of the corresponding maximum.\n\n    .. seealso:: :func:`scipy.ndimage.extrema`\n    '
    dims = numpy.array(input.shape)
    dim_prod = numpy.cumprod([1] + list(dims[:0:-1]))[::-1]
    (minimums, min_positions, maximums, max_positions) = _select(input, labels, index, find_min=True, find_max=True, find_min_positions=True, find_max_positions=True)
    if min_positions.ndim == 0:
        min_positions = min_positions.item()
        max_positions = max_positions.item()
        return (minimums, maximums, tuple(min_positions // dim_prod % dims), tuple(max_positions // dim_prod % dims))
    min_positions = cupy.asnumpy(min_positions)
    max_positions = cupy.asnumpy(max_positions)
    min_positions = [tuple(v) for v in min_positions.reshape(-1, 1) // dim_prod % dims]
    max_positions = [tuple(v) for v in max_positions.reshape(-1, 1) // dim_prod % dims]
    return (minimums, maximums, min_positions, max_positions)

def center_of_mass(input, labels=None, index=None):
    if False:
        for i in range(10):
            print('nop')
    '\n    Calculate the center of mass of the values of an array at labels.\n\n    Args:\n        input (cupy.ndarray): Data from which to calculate center-of-mass. The\n            masses can either be positive or negative.\n        labels (cupy.ndarray, optional): Labels for objects in `input`, as\n            enerated by `ndimage.label`. Only used with `index`. Dimensions\n            must be the same as `input`.\n        index (int or sequence of ints, optional): Labels for which to\n            calculate centers-of-mass. If not specified, all labels greater\n            than zero are used. Only used with `labels`.\n\n    Returns:\n        tuple or list of tuples: Coordinates of centers-of-mass.\n\n    .. seealso:: :func:`scipy.ndimage.center_of_mass`\n    '
    normalizer = sum(input, labels, index)
    grids = cupy.ogrid[[slice(0, i) for i in input.shape]]
    results = [sum(input * grids[dir].astype(float), labels, index) / normalizer for dir in range(input.ndim)]
    is_0dim_array = isinstance(results[0], cupy.ndarray) and results[0].ndim == 0
    if is_0dim_array:
        return tuple((res for res in results))
    return [v for v in cupy.stack(results, axis=-1)]

def labeled_comprehension(input, labels, index, func, out_dtype, default, pass_positions=False):
    if False:
        print('Hello World!')
    'Array resulting from applying ``func`` to each labeled region.\n\n    Roughly equivalent to [func(input[labels == i]) for i in index].\n\n    Sequentially applies an arbitrary function (that works on array_like input)\n    to subsets of an N-D image array specified by `labels` and `index`.\n    The option exists to provide the function with positional parameters as the\n    second argument.\n\n    Args:\n        input (cupy.ndarray): Data from which to select `labels` to process.\n        labels (cupy.ndarray or None):  Labels to objects in `input`. If not\n            None, array must be same shape as `input`. If None, `func` is\n            applied to raveled `input`.\n        index (int, sequence of ints or None): Subset of `labels` to which to\n            apply `func`. If a scalar, a single value is returned. If None,\n            `func` is applied to all non-zero values of `labels`.\n        func (callable): Python function to apply to `labels` from `input`.\n        out_dtype (dtype): Dtype to use for `result`.\n        default (int, float or None): Default return value when a element of\n            `index` does not exist in `labels`.\n        pass_positions (bool, optional): If True, pass linear indices to `func`\n            as a second argument.\n\n    Returns:\n        cupy.ndarray: Result of applying `func` to each of `labels` to `input`\n        in `index`.\n\n    .. seealso:: :func:`scipy.ndimage.labeled_comprehension`\n    '
    as_scalar = cupy.isscalar(index)
    input = cupy.asarray(input)
    if pass_positions:
        positions = cupy.arange(input.size).reshape(input.shape)
    if labels is None:
        if index is not None:
            raise ValueError('index without defined labels')
        if not pass_positions:
            return func(input.ravel())
        else:
            return func(input.ravel(), positions.ravel())
    try:
        (input, labels) = cupy.broadcast_arrays(input, labels)
    except ValueError:
        raise ValueError('input and labels must have the same shape (excepting dimensions with width 1)')
    if index is None:
        if not pass_positions:
            return func(input[labels > 0])
        else:
            return func(input[labels > 0], positions[labels > 0])
    index = cupy.atleast_1d(index)
    if cupy.any(index.astype(labels.dtype).astype(index.dtype) != index):
        raise ValueError('Cannot convert index values from <%s> to <%s> (labels.dtype) without loss of precision' % (index.dtype, labels.dtype))
    index = index.astype(labels.dtype)
    lo = index.min()
    hi = index.max()
    mask = (labels >= lo) & (labels <= hi)
    labels = labels[mask]
    input = input[mask]
    if pass_positions:
        positions = positions[mask]
    label_order = labels.argsort()
    labels = labels[label_order]
    input = input[label_order]
    if pass_positions:
        positions = positions[label_order]
    index_order = index.argsort()
    sorted_index = index[index_order]

    def do_map(inputs, output):
        if False:
            print('Hello World!')
        'labels must be sorted'
        nidx = sorted_index.size
        lo = cupy.searchsorted(labels, sorted_index, side='left')
        hi = cupy.searchsorted(labels, sorted_index, side='right')
        for (i, low, high) in zip(range(nidx), lo, hi):
            if low == high:
                continue
            output[i] = func(*[inp[low:high] for inp in inputs])
    if out_dtype == object:
        temp = {i: default for i in range(index.size)}
    else:
        temp = cupy.empty(index.shape, out_dtype)
        if default is None and temp.dtype.kind in 'fc':
            default = numpy.nan
        temp[:] = default
    if not pass_positions:
        do_map([input], temp)
    else:
        do_map([input, positions], temp)
    if out_dtype == object:
        index_order = cupy.asnumpy(index_order)
        output = [temp[i] for i in index_order.argsort()]
    else:
        output = cupy.zeros(index.shape, out_dtype)
        output[cupy.asnumpy(index_order)] = temp
    if as_scalar:
        output = output[0]
    return output

def histogram(input, min, max, bins, labels=None, index=None):
    if False:
        return 10
    'Calculate the histogram of the values of an array, optionally at labels.\n\n    Histogram calculates the frequency of values in an array within bins\n    determined by `min`, `max`, and `bins`. The `labels` and `index`\n    keywords can limit the scope of the histogram to specified sub-regions\n    within the array.\n\n    Args:\n        input (cupy.ndarray): Data for which to calculate histogram.\n        min (int): Minimum values of range of histogram bins.\n        max (int): Maximum values of range of histogram bins.\n        bins (int): Number of bins.\n        labels (cupy.ndarray, optional): Labels for objects in `input`. If not\n            None, must be same shape as `input`.\n        index (int or sequence of ints, optional): Label or labels for which to\n            calculate histogram. If None, all values where label is greater\n            than zero are used.\n\n    Returns:\n        cupy.ndarray: Histogram counts.\n\n    .. seealso:: :func:`scipy.ndimage.histogram`\n    '
    _bins = cupy.linspace(min, max, bins + 1)

    def _hist(vals):
        if False:
            return 10
        return cupy.histogram(vals, _bins)[0]
    return labeled_comprehension(input, labels, index, _hist, object, None, pass_positions=False)

def value_indices(arr, *, ignore_value=None, adaptive_index_dtype=False):
    if False:
        while True:
            i = 10
    "\n    Find indices of each distinct value in given array.\n\n    Parameters\n    ----------\n    arr : ndarray of ints\n        Array containing integer values.\n    ignore_value : int, optional\n        This value will be ignored in searching the `arr` array. If not\n        given, all values found will be included in output. Default\n        is None.\n    adaptive_index_dtype : bool, optional\n        If ``True``, instead of returning the default CuPy signed integer\n        dtype, the smallest signed integer dtype capable of representing the\n        image coordinate range will be used. This can substantially reduce\n        memory usage and slightly reduce runtime. Note that this optional\n        parameter is not available in the SciPy API.\n\n    Returns\n    -------\n    indices : dictionary\n        A Python dictionary of array indices for each distinct value. The\n        dictionary is keyed by the distinct values, the entries are array\n        index tuples covering all occurrences of the value within the\n        array.\n\n        This dictionary can occupy significant memory, often several times\n        the size of the input array. To help reduce memory overhead, the\n        argument `adaptive_index_dtype` can be set to ``True``.\n\n    Notes\n    -----\n    For a small array with few distinct values, one might use\n    `numpy.unique()` to find all possible values, and ``(arr == val)`` to\n    locate each value within that array. However, for large arrays,\n    with many distinct values, this can become extremely inefficient,\n    as locating each value would require a new search through the entire\n    array. Using this function, there is essentially one search, with\n    the indices saved for all distinct values.\n\n    This is useful when matching a categorical image (e.g. a segmentation\n    or classification) to an associated image of other data, allowing\n    any per-class statistic(s) to then be calculated. Provides a\n    more flexible alternative to functions like ``scipy.ndimage.mean()``\n    and ``scipy.ndimage.variance()``.\n\n    Some other closely related functionality, with different strengths and\n    weaknesses, can also be found in ``scipy.stats.binned_statistic()`` and\n    the `scikit-image <https://scikit-image.org/>`_ function\n    ``skimage.measure.regionprops()``.\n\n    Note for IDL users: this provides functionality equivalent to IDL's\n    REVERSE_INDICES option (as per the IDL documentation for the\n    `HISTOGRAM <https://www.l3harrisgeospatial.com/docs/histogram.html>`_\n    function).\n\n    .. versionadded:: 1.10.0\n\n    See Also\n    --------\n    label, maximum, median, minimum_position, extrema, sum, mean, variance,\n    standard_deviation, cupy.where, cupy.unique\n\n    Examples\n    --------\n    >>> import cupy\n    >>> from cupyx.scipy import ndimage\n    >>> a = cupy.zeros((6, 6), dtype=int)\n    >>> a[2:4, 2:4] = 1\n    >>> a[4, 4] = 1\n    >>> a[:2, :3] = 2\n    >>> a[0, 5] = 3\n    >>> a\n    array([[2, 2, 2, 0, 0, 3],\n           [2, 2, 2, 0, 0, 0],\n           [0, 0, 1, 1, 0, 0],\n           [0, 0, 1, 1, 0, 0],\n           [0, 0, 0, 0, 1, 0],\n           [0, 0, 0, 0, 0, 0]])\n    >>> val_indices = ndimage.value_indices(a)\n\n    The dictionary `val_indices` will have an entry for each distinct\n    value in the input array.\n\n    >>> val_indices.keys()\n    dict_keys([0, 1, 2, 3])\n\n    The entry for each value is an index tuple, locating the elements\n    with that value.\n\n    >>> ndx1 = val_indices[1]\n    >>> ndx1\n    (array([2, 2, 3, 3, 4]), array([2, 3, 2, 3, 4]))\n\n    This can be used to index into the original array, or any other\n    array with the same shape.\n\n    >>> a[ndx1]\n    array([1, 1, 1, 1, 1])\n\n    If the zeros were to be ignored, then the resulting dictionary\n    would no longer have an entry for zero.\n\n    >>> val_indices = ndimage.value_indices(a, ignore_value=0)\n    >>> val_indices.keys()\n    dict_keys([1, 2, 3])\n\n    "
    if arr.dtype.kind not in 'iu':
        raise ValueError("Parameter 'arr' must be an integer array")
    if adaptive_index_dtype:
        raveled_int_type = cupy.min_scalar_type(-(int(arr.size) + 1))
        coord_int_type = cupy.min_scalar_type(-(max(arr.shape) + 1))
    arr1d = arr.reshape(-1)
    counts = cupy.bincount(arr1d)
    isort = cupy.argsort(arr1d, axis=None)
    if adaptive_index_dtype:
        isort = isort.astype(raveled_int_type, copy=False)
    coords = cupy.unravel_index(isort, arr.shape)
    if adaptive_index_dtype:
        coords = tuple((c.astype(coord_int_type, copy=False) for c in coords))
    offset = 0
    out = {}
    counts = cupy.asnumpy(counts)
    for (value, count) in enumerate(counts):
        if count == 0:
            continue
        elif value == ignore_value:
            offset += count
            continue
        out[value] = tuple((c[offset:offset + count] for c in coords))
        offset += count
    return out