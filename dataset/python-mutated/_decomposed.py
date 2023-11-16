import torch
from torch.library import Library, impl
from torch.ao.quantization.utils import determine_qparams, validate_qmin_qmax
from typing import Tuple
quantized_decomposed_lib = Library('quantized_decomposed', 'DEF')
_DTYPE_TO_QVALUE_BOUNDS = {torch.uint8: (0, 255), torch.int8: (-128, 127), torch.int16: (-2 ** 15, 2 ** 15 - 1), torch.int32: (-2 ** 31, 2 ** 31 - 1)}

def _quant_min_max_bounds_check(quant_min, quant_max, dtype):
    if False:
        return 10
    if dtype not in _DTYPE_TO_QVALUE_BOUNDS:
        raise ValueError(f'Unsupported dtype: {dtype}')
    (quant_min_lower_bound, quant_max_upper_bound) = _DTYPE_TO_QVALUE_BOUNDS[dtype]
    assert quant_min >= quant_min_lower_bound, f'quant_min out of bound for dtype, quant_min_lower_bound: {quant_min_lower_bound} quant_min: {quant_min}'
    assert quant_max <= quant_max_upper_bound, f'quant_max out of bound for dtype, quant_max_upper_bound: {quant_max_upper_bound} quant_max: {quant_max}'
quantized_decomposed_lib.define('quantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'quantize_per_tensor', 'CompositeExplicitAutograd')
def quantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    ' Affine quantization for the Tensor using the same quantization parameters to map\n    from floating point to quantized values\n\n    Args:\n       input (torch.Tensor): original float32 or bfloat16 Tensor\n       scale (float): quantization parameter for affine quantization\n       zero_point (int): quantization parameter for affine quantization\n       quant_min (int): minimum quantized value for output Tensor\n       quant_max (int): maximum quantized value for output Tensor\n       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor\n\n    Returns:\n       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters\n       are not stored in the Tensor, we are storing them in function arguments instead\n    '
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)
quantized_decomposed_lib.define('quantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'quantize_per_tensor.tensor', 'CompositeExplicitAutograd')
def quantize_per_tensor_tensor(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        while True:
            i = 10
    ' Affine quantization for the Tensor using the same quantization parameters to map\n    from floating point to quantized values\n    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of\n    scalar values\n    '
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)

@impl(quantized_decomposed_lib, 'quantize_per_tensor.tensor', 'Meta')
def quantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    if False:
        while True:
            i = 10
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    return torch.empty_like(input, dtype=dtype)
quantized_decomposed_lib.define('quantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'quantize_per_tensor.tensor2', 'CompositeExplicitAutograd')
def quantize_per_tensor_tensor2(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: torch.Tensor, quant_max: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if False:
        print('Hello World!')
    ' Affine quantization for the Tensor using the same quantization parameters to map\n    from floating point to quantized values\n    Same as `quantize_per_tensor` but scale and zero_point are Scalar Tensor instead of\n    scalar values\n    '
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    return quantize_per_tensor(input, scale.item(), zero_point.item(), quant_min.item(), quant_max.item(), dtype)

@impl(quantized_decomposed_lib, 'quantize_per_tensor.tensor2', 'Meta')
def quantize_per_tensor_tensor2_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    if False:
        for i in range(10):
            print('nop')
    return quantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype)
quantized_decomposed_lib.define('dequantize_per_tensor(Tensor input, float scale, int zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'dequantize_per_tensor', 'CompositeExplicitAutograd')
def dequantize_per_tensor(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    ' Affine dequantization for the Tensor using the same quantization parameters to map\n    from quantized values to floating point values\n\n    Args:\n       input (torch.Tensor): Tensor with dtype matching `dtype` argument,\n       e.g. (`torch.uint8`), it is a per tensor quantized Tensor if combined with\n       quantization parameters in the argument of this function (scale/zero_point)\n\n       scale (float): quantization parameter for affine quantization\n\n       zero_point (int): quantization parameter for affine quantization\n\n       quant_min (int): minimum quantized value for input Tensor (not used in computation,\n       reserved for pattern matching)\n\n       quant_max (int): maximum quantized value for input Tensor (not used in computation,\n       reserved for pattern matching)\n\n       dtype (torch.dtype): dtype for input Tensor (not used in computation,\n       reserved for pattern matching)\n\n    Returns:\n       dequantized float32 Tensor\n    '
    assert input.dtype == dtype, f'Expecting input to have dtype: {dtype}, but got {input.dtype}'
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return (input.to(torch.float32) - zero_point) * scale
    else:
        raise ValueError(f'Unsupported dtype in dequantize_per_tensor: {dtype}')
quantized_decomposed_lib.define('dequantize_per_tensor.tensor(Tensor input, Tensor scale, Tensor zero_point, int quant_min, int quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'dequantize_per_tensor.tensor', 'CompositeExplicitAutograd')
def dequantize_per_tensor_tensor(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    ' Affine dequantization for the Tensor using the same quantization parameters to map\n    from quantized values to floating point values\n    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of\n    scalar values\n    '
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    return dequantize_per_tensor(input, scale.item(), zero_point.item(), quant_min, quant_max, dtype)

@impl(quantized_decomposed_lib, 'dequantize_per_tensor.tensor', 'Meta')
def dequantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    if False:
        for i in range(10):
            print('nop')
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    assert input.dtype == dtype, f'Expecting input to have dtype: {dtype}'
    if dtype in _DTYPE_TO_QVALUE_BOUNDS:
        return torch.empty_like(input, dtype=torch.float32)
    else:
        raise ValueError(f'Unsupported dtype in dequantize_per_tensor: {dtype}')
quantized_decomposed_lib.define('dequantize_per_tensor.tensor2(Tensor input, Tensor scale, Tensor zero_point, Tensor quant_min, Tensor quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'dequantize_per_tensor.tensor2', 'CompositeExplicitAutograd')
def dequantize_per_tensor_tensor2(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: torch.Tensor, quant_max: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    ' Affine dequantization for the Tensor using the same quantization parameters to map\n    from quantized values to floating point values\n    Same as `dequantize_per_tensor` but scale and zero_point are Scalar Tensor instead of\n    scalar values\n    '
    assert zero_point.numel() == 1, f'Expecting zero_point tensor to be one element, but received : {zero_point.numel()}'
    assert scale.numel() == 1, f'Expecting scale tensor to be one element, but received : {scale.numel()}'
    return dequantize_per_tensor(input, scale.item(), zero_point.item(), quant_min.item(), quant_max.item(), dtype)

@impl(quantized_decomposed_lib, 'dequantize_per_tensor.tensor2', 'Meta')
def dequantize_per_tensor_tensor2_meta(input, scale, zero_point, quant_min, quant_max, dtype):
    if False:
        return 10
    return dequantize_per_tensor_tensor_meta(input, scale, zero_point, quant_min, quant_max, dtype)
quantized_decomposed_lib.define('choose_qparams.tensor(Tensor input, int quant_min, int quant_max, float eps, ScalarType dtype) -> (Tensor, Tensor)')

@impl(quantized_decomposed_lib, 'choose_qparams.tensor', 'CompositeExplicitAutograd')
def choose_qparams_tensor(input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        print('Hello World!')
    ' Given an input Tensor, derive the per tensor affine quantization parameter\n    (scale and zero_point) for target quantized Tensor from the Tensor\n\n    Args:\n       input (torch.Tensor): floating point input Tensor\n       quant_min (int): minimum quantized value for target quantized Tensor\n       quant_max (int): maximum quantized value for target quantized Tensor\n       dtype (torch.dtype): dtype for target quantized Tensor\n\n    Returns:\n       scale (float): quantization parameter for the target quantized Tensor\n       zero_point (int): quantization parameter for the target quantized Tensor\n    '
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    assert dtype in _DTYPE_TO_QVALUE_BOUNDS, f'Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}'
    validate_qmin_qmax(qmin, qmax)
    (min_val, max_val) = torch.aminmax(input)
    return determine_qparams(min_val, max_val, qmin, qmax, dtype, torch.Tensor([eps]), has_customized_qrange=False)
quantized_decomposed_lib.define('choose_qparams_symmetric.tensor(Tensor input, int quant_min, int quant_max, float eps, ScalarType dtype) -> (Tensor, Tensor)')

@impl(quantized_decomposed_lib, 'choose_qparams_symmetric.tensor', 'CompositeExplicitAutograd')
def choose_qparams_symmetric_tensor(input: torch.Tensor, qmin: int, qmax: int, eps: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        while True:
            i = 10
    ' Given an input Tensor, derive the per tensor affine quantization parameter\n    (scale and zero_point) for target quantized Tensor from the Tensor\n\n    Args:\n       input (torch.Tensor): floating point input Tensor\n       quant_min (int): minimum quantized value for target quantized Tensor\n       quant_max (int): maximum quantized value for target quantized Tensor\n       dtype (torch.dtype): dtype for target quantized Tensor\n\n    Returns:\n       scale (float): quantization parameter for the target quantized Tensor\n       zero_point (int): quantization parameter for the target quantized Tensor\n    '
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    assert dtype in _DTYPE_TO_QVALUE_BOUNDS, f'Expecting target dtype to be one of {_DTYPE_TO_QVALUE_BOUNDS.keys()}, but got: {dtype}'
    validate_qmin_qmax(qmin, qmax)
    (min_val, max_val) = torch.aminmax(input)
    return determine_qparams(min_val, max_val, qmin, qmax, dtype, torch.Tensor([eps]), has_customized_qrange=False, qscheme=torch.per_tensor_symmetric)

@impl(quantized_decomposed_lib, 'choose_qparams.tensor', 'Meta')
def choose_qparams_tensor_meta(input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        i = 10
        return i + 15
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    assert quant_min < quant_max, f'Expecting quant_min to be smaller than quant_max but received min:         {quant_min} max: {quant_max}'
    return (torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device))

@impl(quantized_decomposed_lib, 'choose_qparams_symmetric.tensor', 'Meta')
def choose_qparams_symmetric_tensor_meta(input: torch.Tensor, quant_min: int, quant_max: int, eps: float, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    if False:
        print('Hello World!')
    return (torch.empty(1, dtype=torch.double, device=input.device), torch.empty(1, dtype=torch.int64, device=input.device))

def _permute_to_axis_zero(x, axis):
    if False:
        i = 10
        return i + 15
    new_axis_list = list(range(x.dim()))
    new_axis_list[axis] = 0
    new_axis_list[0] = axis
    y = x.permute(tuple(new_axis_list))
    return (y, new_axis_list)
quantized_decomposed_lib.define('quantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, int quant_min, int quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'quantize_per_channel', 'CompositeExplicitAutograd')
def quantize_per_channel(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        return 10
    ' Affine per channel quantization for the Tensor using the same quantization\n    parameters for each channel/axis to map from floating point to quantized values\n\n    Args:\n       input (torch.Tensor): original float32 or bfloat16 Tensor\n       scales (torch.Tensor): a list of scale quantization parameter for\n       affine quantization, one per channel\n       zero_point (torch.Tensor): a list of zero_point quantization parameter for\n       affine quantization, one per channel\n       quant_min (int): minimum quantized value for output Tensor\n       quant_max (int): maximum quantized value for output Tensor\n       dtype (torch.dtype): requested dtype (e.g. torch.uint8) for output Tensor\n\n    Returns:\n       Tensor with requested dtype (e.g. torch.uint8), note the quantization parameters\n       are not stored in the Tensor, we are storing them in function arguments instead\n    '
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    assert axis < input.dim(), f'Expecting axis to be < {input.dim()}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    (input, permute_axis_list) = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input)
    for i in range(input.size(0)):
        res[i] = torch.clamp(torch.round(input[i] * (1.0 / scales[i])) + zero_points[i], quant_min, quant_max)
    out = res.permute(tuple(permute_axis_list))
    return out.to(dtype)

@impl(quantized_decomposed_lib, 'quantize_per_channel', 'Meta')
def quantize_per_channel_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        return 10
    assert input.dtype == torch.float32, f'Expecting input to have dtype torch.float32, but got dtype: {input.dtype}'
    assert axis < input.dim(), f'Expecting axis to be < {input.dim()}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=dtype)
quantized_decomposed_lib.define('dequantize_per_channel(Tensor input, Tensor scales, Tensor zero_points, int axis, int quant_min, int quant_max, ScalarType dtype) -> Tensor')

@impl(quantized_decomposed_lib, 'dequantize_per_channel', 'CompositeExplicitAutograd')
def dequantize_per_channel(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        i = 10
        return i + 15
    ' Affine per channel dequantization for the Tensor using the same quantization\n    parameters for each channel/axis to map from quantized values to floating point values\n\n    Args:\n       input (torch.Tensor): Tensor with dtype matching `dtype` argument,\n       e.g. (`torch.uint8`), it is a per channel quantized Tensor if combined with\n       quantization parameter in the argument of this function (scales/zero_points/axis)\n\n       scales (torch.Tensor): a list of scale quantization parameter for\n       affine quantization, one per channel\n\n       zero_points (torch.Tensor): a list of zero_point quantization parameter for\n       affine quantization, one per channel\n\n       quant_min (int): minimum quantized value for output Tensor (not used in computation,\n       reserved for pattern matching)\n\n       quant_max (int): maximum quantized value for output Tensor (not used in computation,\n       reserved for pattern matching)\n\n       dtype (torch.dtype): requested dtype for output Tensor (not used in computation,\n       reserved for pattern matching)\n\n    Returns:\n       dequantized float32 Tensor\n    '
    assert input.dtype == dtype, f'Expecting input to have dtype {dtype}, but got dtype: {input.dtype}'
    assert axis < input.dim(), f'Expecting axis to be < {input.dim()}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    (input, permute_axis_list) = _permute_to_axis_zero(input, axis)
    res = torch.zeros_like(input, dtype=torch.float32)
    for i in range(input.size(0)):
        res[i] = (input[i].to(torch.float32) - zero_points[i]) * scales[i]
    out = res.permute(tuple(permute_axis_list))
    return out

@impl(quantized_decomposed_lib, 'dequantize_per_channel', 'Meta')
def dequantize_per_channel_meta(input: torch.Tensor, scales: torch.Tensor, zero_points: torch.Tensor, axis: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        while True:
            i = 10
    assert input.dtype == dtype, f'Expecting input to have dtype {dtype}, but got dtype: {input.dtype}'
    assert axis < input.dim(), f'Expecting axis to be < {input.dim()}'
    _quant_min_max_bounds_check(quant_min, quant_max, dtype)
    return torch.empty_like(input, dtype=torch.float32)