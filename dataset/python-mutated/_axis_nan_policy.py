import numpy as np
from functools import wraps
from scipy._lib._docscrape import FunctionDoc, Parameter
from scipy._lib._util import _contains_nan, AxisError, _get_nan
import inspect

def _broadcast_arrays(arrays, axis=None):
    if False:
        print('Hello World!')
    '\n    Broadcast shapes of arrays, ignoring incompatibility of specified axes\n    '
    new_shapes = _broadcast_array_shapes(arrays, axis=axis)
    if axis is None:
        new_shapes = [new_shapes] * len(arrays)
    return [np.broadcast_to(array, new_shape) for (array, new_shape) in zip(arrays, new_shapes)]

def _broadcast_array_shapes(arrays, axis=None):
    if False:
        print('Hello World!')
    '\n    Broadcast shapes of arrays, ignoring incompatibility of specified axes\n    '
    shapes = [np.asarray(arr).shape for arr in arrays]
    return _broadcast_shapes(shapes, axis)

def _broadcast_shapes(shapes, axis=None):
    if False:
        i = 10
        return i + 15
    '\n    Broadcast shapes, ignoring incompatibility of specified axes\n    '
    if not shapes:
        return shapes
    if axis is not None:
        axis = np.atleast_1d(axis)
        axis_int = axis.astype(int)
        if not np.array_equal(axis_int, axis):
            raise AxisError('`axis` must be an integer, a tuple of integers, or `None`.')
        axis = axis_int
    n_dims = max([len(shape) for shape in shapes])
    new_shapes = np.ones((len(shapes), n_dims), dtype=int)
    for (row, shape) in zip(new_shapes, shapes):
        row[len(row) - len(shape):] = shape
    if axis is not None:
        axis[axis < 0] = n_dims + axis[axis < 0]
        axis = np.sort(axis)
        if axis[-1] >= n_dims or axis[0] < 0:
            message = f'`axis` is out of bounds for array of dimension {n_dims}'
            raise AxisError(message)
        if len(np.unique(axis)) != len(axis):
            raise AxisError('`axis` must contain only distinct elements')
        removed_shapes = new_shapes[:, axis]
        new_shapes = np.delete(new_shapes, axis, axis=1)
    new_shape = np.max(new_shapes, axis=0)
    new_shape *= new_shapes.all(axis=0)
    if np.any(~((new_shapes == 1) | (new_shapes == new_shape))):
        raise ValueError('Array shapes are incompatible for broadcasting.')
    if axis is not None:
        new_axis = axis - np.arange(len(axis))
        new_shapes = [tuple(np.insert(new_shape, new_axis, removed_shape)) for removed_shape in removed_shapes]
        return new_shapes
    else:
        return tuple(new_shape)

def _broadcast_array_shapes_remove_axis(arrays, axis=None):
    if False:
        i = 10
        return i + 15
    '\n    Broadcast shapes of arrays, dropping specified axes\n\n    Given a sequence of arrays `arrays` and an integer or tuple `axis`, find\n    the shape of the broadcast result after consuming/dropping `axis`.\n    In other words, return output shape of a typical hypothesis test on\n    `arrays` vectorized along `axis`.\n\n    Examples\n    --------\n    >>> import numpy as np\n    >>> from scipy.stats._axis_nan_policy import _broadcast_array_shapes\n    >>> a = np.zeros((5, 2, 1))\n    >>> b = np.zeros((9, 3))\n    >>> _broadcast_array_shapes((a, b), 1)\n    (5, 3)\n    '
    shapes = [arr.shape for arr in arrays]
    return _broadcast_shapes_remove_axis(shapes, axis)

def _broadcast_shapes_remove_axis(shapes, axis=None):
    if False:
        print('Hello World!')
    '\n    Broadcast shapes, dropping specified axes\n\n    Same as _broadcast_array_shapes, but given a sequence\n    of array shapes `shapes` instead of the arrays themselves.\n    '
    shapes = _broadcast_shapes(shapes, axis)
    shape = shapes[0]
    if axis is not None:
        shape = np.delete(shape, axis)
    return tuple(shape)

def _broadcast_concatenate(arrays, axis):
    if False:
        print('Hello World!')
    'Concatenate arrays along an axis with broadcasting.'
    arrays = _broadcast_arrays(arrays, axis)
    res = np.concatenate(arrays, axis=axis)
    return res

def _remove_nans(samples, paired):
    if False:
        while True:
            i = 10
    'Remove nans from paired or unpaired 1D samples'
    if not paired:
        return [sample[~np.isnan(sample)] for sample in samples]
    nans = np.isnan(samples[0])
    for sample in samples[1:]:
        nans = nans | np.isnan(sample)
    not_nans = ~nans
    return [sample[not_nans] for sample in samples]

def _remove_sentinel(samples, paired, sentinel):
    if False:
        i = 10
        return i + 15
    'Remove sentinel values from paired or unpaired 1D samples'
    if not paired:
        return [sample[sample != sentinel] for sample in samples]
    sentinels = samples[0] == sentinel
    for sample in samples[1:]:
        sentinels = sentinels | (sample == sentinel)
    not_sentinels = ~sentinels
    return [sample[not_sentinels] for sample in samples]

def _masked_arrays_2_sentinel_arrays(samples):
    if False:
        while True:
            i = 10
    has_mask = False
    for sample in samples:
        mask = getattr(sample, 'mask', False)
        has_mask = has_mask or np.any(mask)
    if not has_mask:
        return (samples, None)
    dtype = np.result_type(*samples)
    dtype = dtype if np.issubdtype(dtype, np.number) else np.float64
    for i in range(len(samples)):
        samples[i] = samples[i].astype(dtype, copy=False)
    inexact = np.issubdtype(dtype, np.inexact)
    info = np.finfo if inexact else np.iinfo
    (max_possible, min_possible) = (info(dtype).max, info(dtype).min)
    nextafter = np.nextafter if inexact else lambda x, _: x - 1
    sentinel = max_possible
    while sentinel > min_possible:
        for sample in samples:
            if np.any(sample == sentinel):
                sentinel = nextafter(sentinel, -np.inf)
                break
        else:
            break
    else:
        message = 'This function replaces masked elements with sentinel values, but the data contains all distinct values of this data type. Consider promoting the dtype to `np.float64`.'
        raise ValueError(message)
    out_samples = []
    for sample in samples:
        mask = getattr(sample, 'mask', None)
        if mask is not None:
            mask = np.broadcast_to(mask, sample.shape)
            sample = sample.data.copy() if np.any(mask) else sample.data
            sample = np.asarray(sample)
            sample[mask] = sentinel
        out_samples.append(sample)
    return (out_samples, sentinel)

def _check_empty_inputs(samples, axis):
    if False:
        i = 10
        return i + 15
    '\n    Check for empty sample; return appropriate output for a vectorized hypotest\n    '
    if not any((sample.size == 0 for sample in samples)):
        return None
    output_shape = _broadcast_array_shapes_remove_axis(samples, axis)
    output = np.ones(output_shape) * _get_nan(*samples)
    return output

def _add_reduced_axes(res, reduced_axes, keepdims):
    if False:
        for i in range(10):
            print('nop')
    '\n    Add reduced axes back to all the arrays in the result object\n    if keepdims = True.\n    '
    return [np.expand_dims(output, reduced_axes) for output in res] if keepdims else res
_name = 'axis'
_desc = 'If an int, the axis of the input along which to compute the statistic.\nThe statistic of each axis-slice (e.g. row) of the input will appear in a\ncorresponding element of the output.\nIf ``None``, the input will be raveled before computing the statistic.'.split('\n')

def _get_axis_params(default_axis=0, _name=_name, _desc=_desc):
    if False:
        while True:
            i = 10
    _type = f'int or None, default: {default_axis}'
    _axis_parameter_doc = Parameter(_name, _type, _desc)
    _axis_parameter = inspect.Parameter(_name, inspect.Parameter.KEYWORD_ONLY, default=default_axis)
    return (_axis_parameter_doc, _axis_parameter)
_name = 'nan_policy'
_type = "{'propagate', 'omit', 'raise'}"
_desc = 'Defines how to handle input NaNs.\n\n- ``propagate``: if a NaN is present in the axis slice (e.g. row) along\n  which the  statistic is computed, the corresponding entry of the output\n  will be NaN.\n- ``omit``: NaNs will be omitted when performing the calculation.\n  If insufficient data remains in the axis slice along which the\n  statistic is computed, the corresponding entry of the output will be\n  NaN.\n- ``raise``: if a NaN is present, a ``ValueError`` will be raised.'.split('\n')
_nan_policy_parameter_doc = Parameter(_name, _type, _desc)
_nan_policy_parameter = inspect.Parameter(_name, inspect.Parameter.KEYWORD_ONLY, default='propagate')
_name = 'keepdims'
_type = 'bool, default: False'
_desc = 'If this is set to True, the axes which are reduced are left\nin the result as dimensions with size one. With this option,\nthe result will broadcast correctly against the input array.'.split('\n')
_keepdims_parameter_doc = Parameter(_name, _type, _desc)
_keepdims_parameter = inspect.Parameter(_name, inspect.Parameter.KEYWORD_ONLY, default=False)
_standard_note_addition = '\nBeginning in SciPy 1.9, ``np.matrix`` inputs (not recommended for new\ncode) are converted to ``np.ndarray`` before the calculation is performed. In\nthis case, the output will be a scalar or ``np.ndarray`` of appropriate shape\nrather than a 2D ``np.matrix``. Similarly, while masked elements of masked\narrays are ignored, the output will be a scalar or ``np.ndarray`` rather than a\nmasked array with ``mask=False``.'.split('\n')

def _axis_nan_policy_factory(tuple_to_result, default_axis=0, n_samples=1, paired=False, result_to_tuple=None, too_small=0, n_outputs=2, kwd_samples=[], override=None):
    if False:
        while True:
            i = 10
    "Factory for a wrapper that adds axis/nan_policy params to a function.\n\n    Parameters\n    ----------\n    tuple_to_result : callable\n        Callable that returns an object of the type returned by the function\n        being wrapped (e.g. the namedtuple or dataclass returned by a\n        statistical test) provided the separate components (e.g. statistic,\n        pvalue).\n    default_axis : int, default: 0\n        The default value of the axis argument. Standard is 0 except when\n        backwards compatibility demands otherwise (e.g. `None`).\n    n_samples : int or callable, default: 1\n        The number of data samples accepted by the function\n        (e.g. `mannwhitneyu`), a callable that accepts a dictionary of\n        parameters passed into the function and returns the number of data\n        samples (e.g. `wilcoxon`), or `None` to indicate an arbitrary number\n        of samples (e.g. `kruskal`).\n    paired : {False, True}\n        Whether the function being wrapped treats the samples as paired (i.e.\n        corresponding elements of each sample should be considered as different\n        components of the same sample.)\n    result_to_tuple : callable, optional\n        Function that unpacks the results of the function being wrapped into\n        a tuple. This is essentially the inverse of `tuple_to_result`. Default\n        is `None`, which is appropriate for statistical tests that return a\n        statistic, pvalue tuple (rather than, e.g., a non-iterable datalass).\n    too_small : int or callable, default: 0\n        The largest unnacceptably small sample for the function being wrapped.\n        For example, some functions require samples of size two or more or they\n        raise an error. This argument prevents the error from being raised when\n        input is not 1D and instead places a NaN in the corresponding element\n        of the result. If callable, it must accept a list of samples and a\n        dictionary of keyword arguments passed to the wrapper function as\n        arguments and return a bool indicating weather the samples passed are\n        too small.\n    n_outputs : int or callable, default: 2\n        The number of outputs produced by the function given 1d sample(s). For\n        example, hypothesis tests that return a namedtuple or result object\n        with attributes ``statistic`` and ``pvalue`` use the default\n        ``n_outputs=2``; summary statistics with scalar output use\n        ``n_outputs=1``. Alternatively, may be a callable that accepts a\n        dictionary of arguments passed into the wrapped function and returns\n        the number of outputs corresponding with those arguments.\n    kwd_samples : sequence, default: []\n        The names of keyword parameters that should be treated as samples. For\n        example, `gmean` accepts as its first argument a sample `a` but\n        also `weights` as a fourth, optional keyword argument. In this case, we\n        use `n_samples=1` and kwd_samples=['weights'].\n    override : dict, default: {'vectorization': False, 'nan_propagation': True}\n        Pass a dictionary with ``'vectorization': True`` to ensure that the\n        decorator overrides the function's behavior for multimensional input.\n        Use ``'nan_propagation': False`` to ensure that the decorator does not\n        override the function's behavior for ``nan_policy='propagate'``.\n        (See `scipy.stats.mode`, for example.)\n    "
    temp = override or {}
    override = {'vectorization': False, 'nan_propagation': True}
    override.update(temp)
    if result_to_tuple is None:

        def result_to_tuple(res):
            if False:
                while True:
                    i = 10
            return res
    if not callable(too_small):

        def is_too_small(samples, *ts_args, **ts_kwargs):
            if False:
                print('Hello World!')
            for sample in samples:
                if len(sample) <= too_small:
                    return True
            return False
    else:
        is_too_small = too_small

    def axis_nan_policy_decorator(hypotest_fun_in):
        if False:
            print('Hello World!')

        @wraps(hypotest_fun_in)
        def axis_nan_policy_wrapper(*args, _no_deco=False, **kwds):
            if False:
                print('Hello World!')
            if _no_deco:
                return hypotest_fun_in(*args, **kwds)
            params = list(inspect.signature(hypotest_fun_in).parameters)
            if n_samples is None:
                params = [f'arg{i}' for i in range(len(args))] + params[1:]
            maxarg = np.inf if inspect.getfullargspec(hypotest_fun_in).varargs else len(inspect.getfullargspec(hypotest_fun_in).args)
            if len(args) > maxarg:
                hypotest_fun_in(*args, **kwds)
            d_args = dict(zip(params, args))
            intersection = set(d_args) & set(kwds)
            if intersection:
                hypotest_fun_in(*args, **kwds)
            kwds.update(d_args)
            if callable(n_samples):
                n_samp = n_samples(kwds)
            else:
                n_samp = n_samples or len(args)
            n_out = n_outputs
            if callable(n_out):
                n_out = n_out(kwds)
            kwd_samp = [name for name in kwd_samples if kwds.get(name, None) is not None]
            n_kwd_samp = len(kwd_samp)
            if not kwd_samp:
                hypotest_fun_out = hypotest_fun_in
            else:

                def hypotest_fun_out(*samples, **kwds):
                    if False:
                        for i in range(10):
                            print('nop')
                    new_kwds = dict(zip(kwd_samp, samples[n_samp:]))
                    kwds.update(new_kwds)
                    return hypotest_fun_in(*samples[:n_samp], **kwds)
            try:
                samples = [np.atleast_1d(kwds.pop(param)) for param in params[:n_samp] + kwd_samp]
            except KeyError:
                hypotest_fun_in(*args, **kwds)
            vectorized = True if 'axis' in params else False
            vectorized = vectorized and (not override['vectorization'])
            axis = kwds.pop('axis', default_axis)
            nan_policy = kwds.pop('nan_policy', 'propagate')
            keepdims = kwds.pop('keepdims', False)
            del args
            (samples, sentinel) = _masked_arrays_2_sentinel_arrays(samples)
            reduced_axes = axis
            if axis is None:
                if samples:
                    n_dims = np.max([sample.ndim for sample in samples])
                    reduced_axes = tuple(range(n_dims))
                samples = [np.asarray(sample.ravel()) for sample in samples]
            else:
                samples = _broadcast_arrays(samples, axis=axis)
                axis = np.atleast_1d(axis)
                n_axes = len(axis)
                samples = [np.moveaxis(sample, axis, range(-len(axis), 0)) for sample in samples]
                shapes = [sample.shape for sample in samples]
                new_shapes = [shape[:-n_axes] + (np.prod(shape[-n_axes:]),) for shape in shapes]
                samples = [sample.reshape(new_shape) for (sample, new_shape) in zip(samples, new_shapes)]
            axis = -1
            NaN = _get_nan(*samples)
            ndims = np.array([sample.ndim for sample in samples])
            if np.all(ndims <= 1):
                if nan_policy != 'propagate' or override['nan_propagation']:
                    contains_nan = [_contains_nan(sample, nan_policy)[0] for sample in samples]
                else:
                    contains_nan = [False] * len(samples)
                if any(contains_nan) and (nan_policy == 'propagate' and override['nan_propagation']):
                    res = np.full(n_out, NaN)
                    res = _add_reduced_axes(res, reduced_axes, keepdims)
                    return tuple_to_result(*res)
                if any(contains_nan) and nan_policy == 'omit':
                    samples = _remove_nans(samples, paired)
                if sentinel:
                    samples = _remove_sentinel(samples, paired, sentinel)
                res = hypotest_fun_out(*samples, **kwds)
                res = result_to_tuple(res)
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)
            empty_output = _check_empty_inputs(samples, axis)
            if empty_output is not None:
                res = [empty_output.copy() for i in range(n_out)]
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)
            lengths = np.array([sample.shape[axis] for sample in samples])
            split_indices = np.cumsum(lengths)
            x = _broadcast_concatenate(samples, axis)
            if nan_policy != 'propagate' or override['nan_propagation']:
                (contains_nan, _) = _contains_nan(x, nan_policy)
            else:
                contains_nan = False
            if vectorized and (not contains_nan) and (not sentinel):
                res = hypotest_fun_out(*samples, axis=axis, **kwds)
                res = result_to_tuple(res)
                res = _add_reduced_axes(res, reduced_axes, keepdims)
                return tuple_to_result(*res)
            if contains_nan and nan_policy == 'omit':

                def hypotest_fun(x):
                    if False:
                        return 10
                    samples = np.split(x, split_indices)[:n_samp + n_kwd_samp]
                    samples = _remove_nans(samples, paired)
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples, kwds):
                        return np.full(n_out, NaN)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds))
            elif contains_nan and nan_policy == 'propagate' and override['nan_propagation']:

                def hypotest_fun(x):
                    if False:
                        return 10
                    if np.isnan(x).any():
                        return np.full(n_out, NaN)
                    samples = np.split(x, split_indices)[:n_samp + n_kwd_samp]
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples, kwds):
                        return np.full(n_out, NaN)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds))
            else:

                def hypotest_fun(x):
                    if False:
                        for i in range(10):
                            print('nop')
                    samples = np.split(x, split_indices)[:n_samp + n_kwd_samp]
                    if sentinel:
                        samples = _remove_sentinel(samples, paired, sentinel)
                    if is_too_small(samples, kwds):
                        return np.full(n_out, NaN)
                    return result_to_tuple(hypotest_fun_out(*samples, **kwds))
            x = np.moveaxis(x, axis, 0)
            res = np.apply_along_axis(hypotest_fun, axis=0, arr=x)
            res = _add_reduced_axes(res, reduced_axes, keepdims)
            return tuple_to_result(*res)
        (_axis_parameter_doc, _axis_parameter) = _get_axis_params(default_axis)
        doc = FunctionDoc(axis_nan_policy_wrapper)
        parameter_names = [param.name for param in doc['Parameters']]
        if 'axis' in parameter_names:
            doc['Parameters'][parameter_names.index('axis')] = _axis_parameter_doc
        else:
            doc['Parameters'].append(_axis_parameter_doc)
        if 'nan_policy' in parameter_names:
            doc['Parameters'][parameter_names.index('nan_policy')] = _nan_policy_parameter_doc
        else:
            doc['Parameters'].append(_nan_policy_parameter_doc)
        if 'keepdims' in parameter_names:
            doc['Parameters'][parameter_names.index('keepdims')] = _keepdims_parameter_doc
        else:
            doc['Parameters'].append(_keepdims_parameter_doc)
        doc['Notes'] += _standard_note_addition
        doc = str(doc).split('\n', 1)[1]
        axis_nan_policy_wrapper.__doc__ = str(doc)
        sig = inspect.signature(axis_nan_policy_wrapper)
        parameters = sig.parameters
        parameter_list = list(parameters.values())
        if 'axis' not in parameters:
            parameter_list.append(_axis_parameter)
        if 'nan_policy' not in parameters:
            parameter_list.append(_nan_policy_parameter)
        if 'keepdims' not in parameters:
            parameter_list.append(_keepdims_parameter)
        sig = sig.replace(parameters=parameter_list)
        axis_nan_policy_wrapper.__signature__ = sig
        return axis_nan_policy_wrapper
    return axis_nan_policy_decorator