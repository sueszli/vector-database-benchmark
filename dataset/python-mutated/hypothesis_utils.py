from collections import defaultdict
from collections.abc import Iterable
import numpy as np
import torch
import hypothesis
from functools import reduce
from hypothesis import assume
from hypothesis import settings
from hypothesis import strategies as st
from hypothesis.extra import numpy as stnp
from hypothesis.strategies import SearchStrategy
from torch.testing._internal.common_quantized import _calculate_dynamic_qparams, _calculate_dynamic_per_channel_qparams
_ALL_QINT_TYPES = (torch.quint8, torch.qint8, torch.qint32)
_ENFORCED_ZERO_POINT = defaultdict(lambda : None, {torch.quint8: None, torch.qint8: None, torch.qint32: 0})

def _get_valid_min_max(qparams):
    if False:
        i = 10
        return i + 15
    (scale, zero_point, quantized_type) = qparams
    adjustment = 1 + torch.finfo(torch.float).eps
    _long_type_info = torch.iinfo(torch.long)
    (long_min, long_max) = (_long_type_info.min / adjustment, _long_type_info.max / adjustment)
    min_value = max((long_min - zero_point) * scale, long_min / scale + zero_point)
    max_value = min((long_max - zero_point) * scale, long_max / scale + zero_point)
    return (np.float32(min_value), np.float32(max_value))

def _floats_wrapper(*args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if 'width' in kwargs and hypothesis.version.__version_info__ < (3, 67, 0):
        no_nan_and_inf = ('allow_nan' in kwargs and (not kwargs['allow_nan']) or 'allow_nan' not in kwargs) and ('allow_infinity' in kwargs and (not kwargs['allow_infinity']) or 'allow_infinity' not in kwargs)
        min_and_max_not_specified = len(args) == 0 and 'min_value' not in kwargs and ('max_value' not in kwargs)
        if no_nan_and_inf and min_and_max_not_specified:
            if kwargs['width'] == 16:
                kwargs['min_value'] = torch.finfo(torch.float16).min
                kwargs['max_value'] = torch.finfo(torch.float16).max
            elif kwargs['width'] == 32:
                kwargs['min_value'] = torch.finfo(torch.float32).min
                kwargs['max_value'] = torch.finfo(torch.float32).max
            elif kwargs['width'] == 64:
                kwargs['min_value'] = torch.finfo(torch.float64).min
                kwargs['max_value'] = torch.finfo(torch.float64).max
        kwargs.pop('width')
    return st.floats(*args, **kwargs)

def floats(*args, **kwargs):
    if False:
        while True:
            i = 10
    if 'width' not in kwargs:
        kwargs['width'] = 32
    return _floats_wrapper(*args, **kwargs)
'Hypothesis filter to avoid overflows with quantized tensors.\n\nArgs:\n    tensor: Tensor of floats to filter\n    qparams: Quantization parameters as returned by the `qparams`.\n\nReturns:\n    True\n\nRaises:\n    hypothesis.UnsatisfiedAssumption\n\nNote: This filter is slow. Use it only when filtering of the test cases is\n      absolutely necessary!\n'

def assume_not_overflowing(tensor, qparams):
    if False:
        return 10
    (min_value, max_value) = _get_valid_min_max(qparams)
    assume(tensor.min() >= min_value)
    assume(tensor.max() <= max_value)
    return True
'Strategy for generating the quantization parameters.\n\nArgs:\n    dtypes: quantized data types to sample from.\n    scale_min / scale_max: Min and max scales. If None, set to 1e-3 / 1e3.\n    zero_point_min / zero_point_max: Min and max for the zero point. If None,\n        set to the minimum and maximum of the quantized data type.\n        Note: The min and max are only valid if the zero_point is not enforced\n              by the data type itself.\n\nGenerates:\n    scale: Sampled scale.\n    zero_point: Sampled zero point.\n    quantized_type: Sampled quantized type.\n'

@st.composite
def qparams(draw, dtypes=None, scale_min=None, scale_max=None, zero_point_min=None, zero_point_max=None):
    if False:
        i = 10
        return i + 15
    if dtypes is None:
        dtypes = _ALL_QINT_TYPES
    if not isinstance(dtypes, (list, tuple)):
        dtypes = (dtypes,)
    quantized_type = draw(st.sampled_from(dtypes))
    _type_info = torch.iinfo(quantized_type)
    (qmin, qmax) = (_type_info.min, _type_info.max)
    _zp_enforced = _ENFORCED_ZERO_POINT[quantized_type]
    if _zp_enforced is not None:
        zero_point = _zp_enforced
    else:
        _zp_min = qmin if zero_point_min is None else zero_point_min
        _zp_max = qmax if zero_point_max is None else zero_point_max
        zero_point = draw(st.integers(min_value=_zp_min, max_value=_zp_max))
    if scale_min is None:
        scale_min = torch.finfo(torch.float).eps
    if scale_max is None:
        scale_max = torch.finfo(torch.float).max
    scale = draw(floats(min_value=scale_min, max_value=scale_max, width=32))
    return (scale, zero_point, quantized_type)
'Strategy to create different shapes.\nArgs:\n    min_dims / max_dims: minimum and maximum rank.\n    min_side / max_side: minimum and maximum dimensions per rank.\n\nGenerates:\n    Possible shapes for a tensor, constrained to the rank and dimensionality.\n\nExample:\n    # Generates 3D and 4D tensors.\n    @given(Q = qtensor(shapes=array_shapes(min_dims=3, max_dims=4))\n    some_test(self, Q):...\n'

@st.composite
def array_shapes(draw, min_dims=1, max_dims=None, min_side=1, max_side=None, max_numel=None):
    if False:
        for i in range(10):
            print('nop')
    'Return a strategy for array shapes (tuples of int >= 1).'
    assert min_dims < 32
    if max_dims is None:
        max_dims = min(min_dims + 2, 32)
    assert max_dims < 32
    if max_side is None:
        max_side = min_side + 5
    candidate = st.lists(st.integers(min_side, max_side), min_size=min_dims, max_size=max_dims)
    if max_numel is not None:
        candidate = candidate.filter(lambda x: reduce(int.__mul__, x, 1) <= max_numel)
    return draw(candidate.map(tuple))
'Strategy for generating test cases for tensors.\nThe resulting tensor is in float32 format.\n\nArgs:\n    shapes: Shapes under test for the tensor. Could be either a hypothesis\n            strategy, or an iterable of different shapes to sample from.\n    elements: Elements to generate from for the returned data type.\n              If None, the strategy resolves to float within range [-1e6, 1e6].\n    qparams: Instance of the qparams strategy. This is used to filter the tensor\n             such that the overflow would not happen.\n\nGenerates:\n    X: Tensor of type float32. Note that NaN and +/-inf is not included.\n    qparams: (If `qparams` arg is set) Quantization parameters for X.\n        The returned parameters are `(scale, zero_point, quantization_type)`.\n        (If `qparams` arg is None), returns None.\n'

@st.composite
def tensor(draw, shapes=None, elements=None, qparams=None, dtype=np.float32):
    if False:
        i = 10
        return i + 15
    if isinstance(shapes, SearchStrategy):
        _shape = draw(shapes)
    else:
        _shape = draw(st.sampled_from(shapes))
    if qparams is None:
        if elements is None:
            elements = floats(-1000000.0, 1000000.0, allow_nan=False, width=32)
        X = draw(stnp.arrays(dtype=dtype, elements=elements, shape=_shape))
        assume(not (np.isnan(X).any() or np.isinf(X).any()))
        return (X, None)
    qparams = draw(qparams)
    if elements is None:
        (min_value, max_value) = _get_valid_min_max(qparams)
        elements = floats(min_value, max_value, allow_infinity=False, allow_nan=False, width=32)
    X = draw(stnp.arrays(dtype=dtype, elements=elements, shape=_shape))
    (scale, zp) = _calculate_dynamic_qparams(X, qparams[2])
    enforced_zp = _ENFORCED_ZERO_POINT.get(qparams[2], None)
    if enforced_zp is not None:
        zp = enforced_zp
    return (X, (scale, zp, qparams[2]))

@st.composite
def per_channel_tensor(draw, shapes=None, elements=None, qparams=None):
    if False:
        i = 10
        return i + 15
    if isinstance(shapes, SearchStrategy):
        _shape = draw(shapes)
    else:
        _shape = draw(st.sampled_from(shapes))
    if qparams is None:
        if elements is None:
            elements = floats(-1000000.0, 1000000.0, allow_nan=False, width=32)
        X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))
        assume(not (np.isnan(X).any() or np.isinf(X).any()))
        return (X, None)
    qparams = draw(qparams)
    if elements is None:
        (min_value, max_value) = _get_valid_min_max(qparams)
        elements = floats(min_value, max_value, allow_infinity=False, allow_nan=False, width=32)
    X = draw(stnp.arrays(dtype=np.float32, elements=elements, shape=_shape))
    (scale, zp) = _calculate_dynamic_per_channel_qparams(X, qparams[2])
    enforced_zp = _ENFORCED_ZERO_POINT.get(qparams[2], None)
    if enforced_zp is not None:
        zp = enforced_zp
    axis = int(np.random.randint(0, X.ndim, 1))
    permute_axes = np.arange(X.ndim)
    permute_axes[0] = axis
    permute_axes[axis] = 0
    X = np.transpose(X, permute_axes)
    return (X, (scale, zp, axis, qparams[2]))
'Strategy for generating test cases for tensors used in Conv.\nThe resulting tensors is in float32 format.\n\nArgs:\n    spatial_dim: Spatial Dim for feature maps. If given as an iterable, randomly\n                 picks one from the pool to make it the spatial dimension\n    batch_size_range: Range to generate `batch_size`.\n                      Must be tuple of `(min, max)`.\n    input_channels_per_group_range:\n        Range to generate `input_channels_per_group`.\n        Must be tuple of `(min, max)`.\n    output_channels_per_group_range:\n        Range to generate `output_channels_per_group`.\n        Must be tuple of `(min, max)`.\n    feature_map_range: Range to generate feature map size for each spatial_dim.\n                       Must be tuple of `(min, max)`.\n    kernel_range: Range to generate kernel size for each spatial_dim. Must be\n                  tuple of `(min, max)`.\n    max_groups: Maximum number of groups to generate.\n    elements: Elements to generate from for the returned data type.\n              If None, the strategy resolves to float within range [-1e6, 1e6].\n    qparams: Strategy for quantization parameters. for X, w, and b.\n             Could be either a single strategy (used for all) or a list of\n             three strategies for X, w, b.\nGenerates:\n    (X, W, b, g): Tensors of type `float32` of the following drawen shapes:\n        X: (`batch_size, input_channels, H, W`)\n        W: (`output_channels, input_channels_per_group) + kernel_shape\n        b: `(output_channels,)`\n        groups: Number of groups the input is divided into\nNote: X, W, b are tuples of (Tensor, qparams), where qparams could be either\n      None or (scale, zero_point, quantized_type)\n\n\nExample:\n    @given(tensor_conv(\n        spatial_dim=2,\n        batch_size_range=(1, 3),\n        input_channels_per_group_range=(1, 7),\n        output_channels_per_group_range=(1, 7),\n        feature_map_range=(6, 12),\n        kernel_range=(3, 5),\n        max_groups=4,\n        elements=st.floats(-1.0, 1.0),\n        qparams=qparams()\n    ))\n'

@st.composite
def tensor_conv(draw, spatial_dim=2, batch_size_range=(1, 4), input_channels_per_group_range=(3, 7), output_channels_per_group_range=(3, 7), feature_map_range=(6, 12), kernel_range=(3, 7), max_groups=1, can_be_transposed=False, elements=None, qparams=None):
    if False:
        for i in range(10):
            print('nop')
    batch_size = draw(st.integers(*batch_size_range))
    input_channels_per_group = draw(st.integers(*input_channels_per_group_range))
    output_channels_per_group = draw(st.integers(*output_channels_per_group_range))
    groups = draw(st.integers(1, max_groups))
    input_channels = input_channels_per_group * groups
    output_channels = output_channels_per_group * groups
    if isinstance(spatial_dim, Iterable):
        spatial_dim = draw(st.sampled_from(spatial_dim))
    feature_map_shape = []
    for i in range(spatial_dim):
        feature_map_shape.append(draw(st.integers(*feature_map_range)))
    kernels = []
    for i in range(spatial_dim):
        kernels.append(draw(st.integers(*kernel_range)))
    tr = False
    weight_shape = (output_channels, input_channels_per_group) + tuple(kernels)
    bias_shape = output_channels
    if can_be_transposed:
        tr = draw(st.booleans())
        if tr:
            weight_shape = (input_channels, output_channels_per_group) + tuple(kernels)
            bias_shape = output_channels
    if qparams is not None:
        if isinstance(qparams, (list, tuple)):
            assert len(qparams) == 3, 'Need 3 qparams for X, w, b'
        else:
            qparams = [qparams] * 3
    X = draw(tensor(shapes=((batch_size, input_channels) + tuple(feature_map_shape),), elements=elements, qparams=qparams[0]))
    W = draw(tensor(shapes=(weight_shape,), elements=elements, qparams=qparams[1]))
    b = draw(tensor(shapes=(bias_shape,), elements=elements, qparams=qparams[2]))
    return (X, W, b, groups, tr)
hypothesis_version = hypothesis.version.__version_info__
current_settings = settings._profiles[settings._current_profile].__dict__
current_settings['deadline'] = None
if hypothesis_version >= (3, 16, 0) and hypothesis_version < (5, 0, 0):
    current_settings['timeout'] = hypothesis.unlimited

def assert_deadline_disabled():
    if False:
        i = 10
        return i + 15
    if hypothesis_version < (3, 27, 0):
        import warnings
        warning_message = f'Your version of hypothesis is outdated. To avoid `DeadlineExceeded` errors, please update. Current hypothesis version: {hypothesis.__version__}'
        warnings.warn(warning_message)
    else:
        assert settings().deadline is None