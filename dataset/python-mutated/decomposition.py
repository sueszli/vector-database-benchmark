import functools
import logging
import math
import sys
import typing
from typing import Optional
import torch
import torch._decomp as decomp
import torch._prims_common as utils
import torch.ao.quantization.fx._decomposed
from torch._decomp import core_aten_decompositions, get_decompositions, remove_decompositions
from torch._decomp.decompositions import _grid_sampler_2d as decomp_grid_sampler_2d, pw_cast_for_opmath
from torch._decomp.decompositions_for_rng import extra_random_decomps
from torch._higher_order_ops.out_dtype import out_dtype
from torch._prims_common import type_to_dtype
from . import config, inductor_prims
log = logging.getLogger(__name__)
aten = torch.ops.aten
prims = torch.ops.prims
quantized_decomposed = torch.ops.quantized_decomposed
inductor_decompositions = get_decompositions([aten._adaptive_avg_pool2d_backward, aten.arange, aten.bitwise_and_, aten.bitwise_or_, aten.clamp_min_, aten.dist, aten.empty_like, aten.flip, aten.gelu, aten.hardtanh, aten.index_select, aten.lcm, aten.leaky_relu, aten.linalg_vector_norm, aten._log_softmax, aten.max_pool2d_with_indices_backward, aten._native_batch_norm_legit, aten._native_batch_norm_legit_functional, aten._native_batch_norm_legit_no_training, aten.native_batch_norm, aten.native_group_norm, aten.native_layer_norm, aten._softmax, aten.sin_, aten.sqrt_, out_dtype, aten._to_copy, aten.tril_indices, aten.triu_indices, aten.upsample_bilinear2d.vec])
decompositions = {**core_aten_decompositions(), **inductor_decompositions}
decomps_to_exclude = [aten._unsafe_index, aten._scaled_dot_product_flash_attention.default, aten.clamp_max, aten.clamp_min, aten.glu, aten.split.Tensor, aten.squeeze, aten.sum, aten.unbind]
remove_decompositions(decompositions, decomps_to_exclude)

def register_decomposition(ops):
    if False:
        print('Hello World!')
    for op in [ops] if callable(ops) else ops:
        if op in decompositions:
            log.warning('duplicate decomp: %s', ops)
    return decomp.register_decomposition(ops, decompositions)

@register_decomposition([aten._assert_async.msg])
def assert_async_msg_decomp(tensor, msg):
    if False:
        return 10
    return

@register_decomposition([aten._functional_assert_async.msg])
def functional_assert_async_msg_decomp(tensor, msg):
    if False:
        return 10
    return

@register_decomposition([aten.sym_constrain_range_for_size.default])
def sym_constrain_range_for_size(symbol, *, min=None, max=None):
    if False:
        return 10
    return

@register_decomposition([aten.clamp])
@pw_cast_for_opmath
def clamp(x, min=None, max=None):
    if False:
        return 10
    if min is not None:
        x = x.clamp_min(min)
    if max is not None:
        x = x.clamp_max(max)
    return x

@register_decomposition([aten.full])
def full(size, fill_value, **kwargs):
    if False:
        print('Hello World!')
    dtype = kwargs.get('dtype')
    if dtype is None:
        kwargs['dtype'] = type_to_dtype(type(fill_value))
        return aten.full(size, fill_value, **kwargs)
    return NotImplemented

@register_decomposition([aten.empty_permuted.default])
def empty_permuted(size, physical_layout, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    perm = [0] * len(size)
    for (p, l) in enumerate(physical_layout):
        perm[l] = p
    return torch.empty([size[l] for l in physical_layout], **kwargs).permute(perm)

@register_decomposition([aten.convolution_backward])
def convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, output_mask):
    if False:
        print('Hello World!')
    if not output_mask[2] or grad_output.device.type != 'cuda':
        return NotImplemented
    grad_bias = aten.sum(grad_output, [0] + list(range(2, grad_output.dim())))
    (grad_inp, grad_weight, _) = aten.convolution_backward(grad_output, input, weight, bias_sizes, stride, padding, dilation, transposed, output_padding, groups, [output_mask[0], output_mask[1], False])
    return (grad_inp, grad_weight, grad_bias)

@register_decomposition([aten.log2])
def log2(x):
    if False:
        while True:
            i = 10
    return torch.log(x) * (1.0 / math.log(2.0))

@register_decomposition([aten.round.decimals])
def round_dec(x, decimals=0):
    if False:
        return 10
    ten_pow_decimals = 10.0 ** decimals
    return aten.round(x * ten_pow_decimals) * (1.0 / ten_pow_decimals)

@register_decomposition([aten.bmm])
@pw_cast_for_opmath
def bmm(self, batch2):
    if False:
        while True:
            i = 10
    if config.coordinate_descent_tuning:
        if self.shape[1] == 1:
            out = (self.unsqueeze(-1) * batch2.unsqueeze(1)).sum(dim=2)
            return out
    if self.device.type == 'cpu':
        if self.size(1) == 1 and batch2.size(-1) == 1:
            return torch.sum(self.squeeze(1) * batch2.squeeze(-1), dim=1, keepdim=True).unsqueeze(1)
    return NotImplemented

@register_decomposition([aten.addmm])
@pw_cast_for_opmath
def addmm(self, mat1, mat2, beta=1, alpha=1):
    if False:
        for i in range(10):
            print('nop')
    if self.device.type == 'cpu':
        if mat1.size(0) == 1 and mat2.size(-1) == 1:
            out = torch.sum(mat1.squeeze(0) * mat2.squeeze(-1), dim=0, keepdim=True).unsqueeze(0)
            return alpha * out + beta * self
        if mat1.size(0) == 1 and mat2.size(0) <= 16 and (mat2.size(1) <= 16):
            out = (mat1.T * mat2).sum(dim=0, keepdim=True)
            return alpha * out + beta * self
    return NotImplemented

@register_decomposition([aten.mm])
@pw_cast_for_opmath
def mm(self, input2):
    if False:
        i = 10
        return i + 15
    if config.coordinate_descent_tuning:
        if self.shape[0] == 1 or input2.shape[1] == 1:
            return (self.unsqueeze(2) * input2.unsqueeze(0)).sum(dim=1)
    if self.device.type == 'cpu':
        if self.size(-1) == 1 and self.size(0) > 0 and (input2.size(0) == 1) and (self.dtype == input2.dtype) and (torch.numel(self) + torch.numel(input2) <= 32):
            return torch.cat([self[i, :] * input2 for i in range(self.size(0))])
        if self.size(0) == 1 and input2.size(-1) == 1:
            return torch.sum(self.squeeze(0) * input2.squeeze(-1), dim=0, keepdim=True).unsqueeze(0)
    return NotImplemented

@register_decomposition([aten.cat.default])
def cat(tensors, dim=0):
    if False:
        print('Hello World!')

    def non_empty_tensor(x):
        if False:
            i = 10
            return i + 15
        return len(x.shape) > 1 or x.shape[0] > 0
    filtered_tensors = list(filter(non_empty_tensor, tensors))
    if len(filtered_tensors) == 1:
        return filtered_tensors[0].clone()
    elif 1 < len(filtered_tensors) < len(tensors):
        return aten.cat.default(filtered_tensors, dim)
    return NotImplemented

@register_decomposition([aten.angle])
def angle(x):
    if False:
        for i in range(10):
            print('nop')
    if x.is_complex():
        return torch.where(torch.isnan(x.real), float('nan'), torch.atan2(x.imag, x.real))
    else:
        ret = torch.where(x < 0, math.pi, 0.0)
        nan = torch.where(torch.isnan(x), float('nan'), 0.0)
        return ret + nan

@register_decomposition([aten.add])
def add(x, y, *, alpha=None):
    if False:
        i = 10
        return i + 15
    x_is_complex_tensor = torch.is_tensor(x) and x.is_complex()
    y_is_complex_tensor = torch.is_tensor(y) and y.is_complex()
    if not x_is_complex_tensor or not y_is_complex_tensor:
        return NotImplemented
    z = y
    if alpha is not None:
        z = alpha * y
    complex_type = torch.promote_types(x.dtype, y.dtype)
    return (x.view(x.real.dtype) + z.view(y.real.dtype)).view(complex_type)

@register_decomposition([aten.conj_physical])
def conj_physical(self):
    if False:
        while True:
            i = 10
    assert not self.is_complex(), 'TODO: implement this'
    return self

@register_decomposition([aten.lift, aten.detach_])
def lift(self):
    if False:
        print('Hello World!')
    return self

@register_decomposition([aten.bernoulli.default])
def bernoulli(self, *, generator=None):
    if False:
        i = 10
        return i + 15
    assert generator is None
    return torch.rand_like(self, dtype=torch.float32) < self

@register_decomposition([aten.fmin, prims.fmin])
def fmin(self, other):
    if False:
        print('Hello World!')
    return torch.where(torch.isnan(other) | (other > self), self, other)

@register_decomposition([aten.fmax, prims.fmax])
def fmax(self, other):
    if False:
        for i in range(10):
            print('nop')
    return torch.where(torch.isnan(other) | (other < self), self, other)

@register_decomposition(aten.amax)
def amax(self, dim=None, keepdim=False):
    if False:
        i = 10
        return i + 15
    if self.dtype == torch.bool:
        return torch.any(self, dim=dim, keepdim=keepdim)
    return NotImplemented

@register_decomposition(aten.amin)
def amin(self, dim=None, keepdim=False):
    if False:
        i = 10
        return i + 15
    if self.dtype == torch.bool:
        return torch.all(self, dim=dim, keepdim=keepdim)
    return NotImplemented

@register_decomposition([aten.narrow_copy])
def narrow_copy(self, dim, start, length):
    if False:
        while True:
            i = 10
    return torch.narrow(self, dim, start, length).clone()

@register_decomposition([aten.expand_copy])
def expand_copy(self, size, *, implicit=False):
    if False:
        print('Hello World!')
    return aten.expand(self, size, implicit=implicit).clone()

@register_decomposition([aten.view_copy.default])
def view_copy_default(self, size):
    if False:
        return 10
    return aten.view(self, size).clone()

@register_decomposition([aten.view_copy.dtype])
def view_copy_dtype(self, dtype):
    if False:
        i = 10
        return i + 15
    return self.to(dtype).clone()

def get_like_layout(tensor: torch.Tensor, memory_format: Optional[torch.memory_format]) -> torch.memory_format:
    if False:
        while True:
            i = 10
    if memory_format is torch.preserve_format or memory_format is None:
        return utils.suggest_memory_format(tensor)
    else:
        return memory_format

@register_decomposition(aten.rand_like)
def rand_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    if False:
        i = 10
        return i + 15
    return torch.rand([*self.size()], dtype=dtype or self.dtype, device=device or self.device, **kwargs).to(memory_format=get_like_layout(self, memory_format))

@register_decomposition(aten.randn_like)
def randn_like(self, *, dtype=None, device=None, memory_format=None, **kwargs):
    if False:
        i = 10
        return i + 15
    return torch.randn([*self.size()], dtype=dtype or self.dtype, device=device or self.device, **kwargs).to(memory_format=get_like_layout(self, memory_format))

@register_decomposition(aten.full_like)
def full_like(self, fill_value, *, dtype=None, layout=None, device=None, pin_memory=False, requires_grad=False, memory_format=torch.preserve_format):
    if False:
        print('Hello World!')
    return torch.full([*self.size()], fill_value, dtype=dtype or self.dtype, layout=layout or self.layout, device=device or self.device, requires_grad=requires_grad).to(memory_format=get_like_layout(self, memory_format))

@register_decomposition(aten.randint_like.default)
def randint_like(self, high, *, dtype=None, device=None, memory_format=None, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return aten.randint.low(0, high, [*self.size()], dtype=dtype or self.dtype, device=device or self.device, **kwargs).to(memory_format=get_like_layout(self, memory_format))

@register_decomposition(aten.randint_like.low_dtype)
def randint_like_low(self, low, high, *, dtype=None, device=None, memory_format=None, **kwargs):
    if False:
        print('Hello World!')
    return aten.randint.low(low, high, [*self.size()], dtype=dtype or self.dtype, device=device or self.device, **kwargs).to(memory_format=get_like_layout(self, memory_format))

@register_decomposition(aten.randint.default)
def randint(high, size, **kwargs):
    if False:
        while True:
            i = 10
    return aten.randint.low(0, high, size, **kwargs)

@register_decomposition(quantized_decomposed.quantize_per_tensor.default)
def quantize_per_tensor_default_decomp_impl(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        return 10
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)

@register_decomposition(quantized_decomposed.dequantize_per_tensor.default)
def dequantize_per_tensor_default_decomp_impl(input: torch.Tensor, scale: float, zero_point: int, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    return (input.to(torch.float32) - zero_point) * scale

@register_decomposition(quantized_decomposed.quantize_per_tensor.tensor)
def quantize_per_tensor_tensor_decomp_impl(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        print('Hello World!')
    if input.dtype == torch.bfloat16:
        input = input.to(torch.float32)
    inv_scale = 1.0 / scale
    return torch.clamp(torch.round(input * inv_scale) + zero_point, quant_min, quant_max).to(dtype)

@register_decomposition(quantized_decomposed.dequantize_per_tensor.tensor)
def dequantize_per_tensor_tensor_decomp_impl(input: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor, quant_min: int, quant_max: int, dtype: torch.dtype) -> torch.Tensor:
    if False:
        return 10
    return (input.to(torch.float32) - zero_point) * scale

@register_decomposition(torch.ops.quantized.embedding_bag_byte_unpack)
def q_embedding_bag_byte_unpack_decomp(packed):
    if False:
        return 10

    def bitcast_u8_to_f32(u8):
        if False:
            print('Hello World!')
        (x, y, z, w) = (u8[..., n].to(torch.int32) for n in (0, 1, 2, 3))
        if sys.byteorder == 'little':
            return (x + (y << 8) + (z << 16) + (w << 24)).view(torch.float32)[..., None]
        else:
            return ((x << 24) + (y << 16) + (z << 8) + w).view(torch.float32)[..., None]
    scales = bitcast_u8_to_f32(packed[..., -8:-4])
    offsets = bitcast_u8_to_f32(packed[..., -4:])
    return packed[..., :-8].to(torch.float32) * scales + offsets

@register_decomposition([aten.grid_sampler_2d])
@pw_cast_for_opmath
def grid_sampler_2d(a: torch.Tensor, grid: torch.Tensor, interpolation_mode: int=0, padding_mode: int=0, align_corners: bool=False) -> torch.Tensor:
    if False:
        for i in range(10):
            print('nop')
    _expand_grid = not (a.device == torch.device('cpu') and interpolation_mode == 0 and a.is_contiguous(memory_format=torch.contiguous_format))
    output = decomp_grid_sampler_2d(a, grid=grid, interpolation_mode=interpolation_mode, padding_mode=padding_mode, align_corners=align_corners, _expand_grid=_expand_grid)
    return output

@register_decomposition(aten._foreach_addcmul.Scalar)
def _foreach_addcmul_scalar(self, left_tensors, right_tensors, scalar=1):
    if False:
        i = 10
        return i + 15
    return aten._foreach_add.List(self, aten._foreach_mul.List(left_tensors, right_tensors), alpha=scalar)

@register_decomposition(aten._foreach_addcdiv.Scalar)
def _foreach_addcdiv_scalar(self, left_tensors, right_tensors, scalar=1):
    if False:
        return 10
    return aten._foreach_add.List(self, aten._foreach_div.List(left_tensors, right_tensors), alpha=scalar)

@register_decomposition(aten._foreach_lerp.Scalar)
def _foreach_lerp_scalar(start_tensors, end_tensors, weight):
    if False:
        for i in range(10):
            print('nop')
    return aten._foreach_add.List(start_tensors, aten._foreach_mul.Scalar(aten._foreach_sub.List(end_tensors, start_tensors), weight))

@aten.miopen_batch_norm.default.py_impl(torch._C.DispatchKey.Autograd)
@register_decomposition(aten.miopen_batch_norm)
def miopen_batch_norm(input: torch.Tensor, weight: torch.Tensor, bias: typing.Optional[torch.Tensor], running_mean: typing.Optional[torch.Tensor], running_var: typing.Optional[torch.Tensor], training: bool, exponential_average_factor: float, epsilon: float):
    if False:
        while True:
            i = 10
    (a, b, c) = aten.native_batch_norm(input, weight, bias, running_mean, running_var, training, exponential_average_factor, epsilon)
    if training:
        return (a, b, c)
    return (a, weight.new_zeros((0,)), weight.new_zeros((0,)))

@functools.lru_cache(None)
def fast_random_decomps():
    if False:
        while True:
            i = 10
    return {**decompositions, **extra_random_decomps}

def select_decomp_table():
    if False:
        print('Hello World!')
    'decomps can change based on config'
    if config.fallback_random:
        return decompositions
    return fast_random_decomps()

@register_decomposition(aten.masked_scatter)
def masked_scatter(self, mask, source):
    if False:
        while True:
            i = 10
    if self.device.type == 'cuda':
        (self, mask) = aten.broadcast_tensors([self, mask])
        source_idx = mask.reshape(-1).cumsum(0) - 1
        return inductor_prims.masked_scatter_with_index(self, mask, source_idx, source)
    return NotImplemented