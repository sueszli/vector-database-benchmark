import unittest
from collections.abc import Sequence
from functools import partial
from typing import List
import numpy as np
import torch
from torch.testing import make_tensor
from torch.testing._internal.common_device_type import tol, toleranceOverride
from torch.testing._internal.common_dtype import all_types_and, all_types_and_complex_and, complex_types, floating_and_complex_types_and, floating_types_and, integral_types
from torch.testing._internal.opinfo.core import DecorateInfo, gradcheck_wrapper_masked_operation, gradcheck_wrapper_masked_pointwise_operation, M, OpInfo, ReductionOpInfo, S, sample_inputs_reduction, SampleInput
from torch.testing._internal.opinfo.utils import prod_numpy, reference_reduction_numpy

def sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, with_dtype=False, use_zero_dimensions=True, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = [((S,), (0,)), ((S, S), (0,)), ((S, S), (1,)), ((S, S), (-1,)), ((S, M, S), (2,)), *([((S, 0, 0), (-1,))] if use_zero_dimensions else [])]
    kwargs = dict(dtype=torch.float64) if with_dtype else None
    if torch.device(device).type != 'xla':
        cases.append(((), (0,)))
    return (SampleInput(make_arg(shape), args=dim, kwargs=kwargs) for (shape, dim) in cases)

def _generate_masked_op_mask(input_shape, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=torch.bool, device=device, requires_grad=False)
    yield None
    yield make_arg(input_shape)
    if len(input_shape) > 2:
        yield make_arg(input_shape[:-1] + (1,))
        yield make_arg(input_shape[:1] + (1,) + input_shape[2:])
        yield make_arg((1,) + input_shape[1:])
        yield make_arg(input_shape[1:])
        yield make_arg(input_shape[-1:])

def sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    'Sample inputs for masked reduction operators.\n\n    Masked reduction operator is a reduction operator with trailing\n    mask optional argument. A mask is a bool tensor with the same\n    shape as input or a shape that is broadcastable to input shape.\n    '
    kwargs['supports_multiple_dims'] = op_info.supports_multiple_dims
    for sample_input in sample_inputs_reduction(op_info, device, dtype, requires_grad, **kwargs):
        for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
            (sample_input_args, sample_input_kwargs) = (sample_input.args, dict(mask=mask, **sample_input.kwargs))
            yield SampleInput(sample_input.input.detach().requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)
            if not requires_grad and dtype.is_floating_point and (sample_input.input.ndim == 2) and (mask is not None) and (mask.shape == sample_input.input.shape):
                for v in [torch.inf, -torch.inf, torch.nan]:
                    t = sample_input.input.detach()
                    t.diagonal(0, -2, -1).fill_(v)
                    yield SampleInput(t.requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)

def sample_inputs_sparse_coo_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    'Sample inputs for masked reduction operators that support inputs\n    with sparse coo layouts.\n    '
    if op_info.supports_sparse:
        op_name = op_info.name.replace('masked.', '')
        for sample_input in sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
            mask = sample_input.kwargs.get('mask')
            if mask is not None:
                sample_input_kwargs = sample_input.kwargs.copy()
                sample_input_kwargs.update(mask=mask.to_sparse())
                yield SampleInput(sample_input.input.to_sparse(), args=sample_input.args, kwargs=sample_input_kwargs)
            else:
                if op_name in {'prod', 'amax', 'amin'}:
                    continue
                yield SampleInput(sample_input.input.to_sparse(), args=sample_input.args, kwargs=sample_input.kwargs)

def sample_inputs_sparse_csr_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    'Sample inputs for masked reduction operators that support inputs\n    with sparse csr layouts.\n    '
    if op_info.supports_sparse_csr:
        op_name = op_info.name.replace('masked.', '')
        for sample_input in sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
            if not (sample_input.input.ndim == 2 and sample_input.kwargs.get('keepdim')):
                continue
            mask = sample_input.kwargs.get('mask')
            if mask is not None:
                sample_input_kwargs = sample_input.kwargs.copy()
                sample_input_kwargs.update(mask=mask.to_sparse_csr())
                new_sample = SampleInput(sample_input.input.to_sparse_csr(), args=sample_input.args, kwargs=sample_input_kwargs)
            else:
                if op_name in ['prod', 'amax', 'amin', 'mean']:
                    continue
                new_sample = SampleInput(sample_input.input.to_sparse_csr(), args=sample_input.args, kwargs=sample_input.kwargs)
            yield new_sample
            if sample_input.kwargs['dim'] == 0:
                sample_input_kwargs = new_sample.kwargs.copy()
                sample_input_kwargs.update(dim=1)
                yield SampleInput(new_sample.input.clone(), args=sample_input.args, kwargs=sample_input_kwargs)

def sample_inputs_masked_norm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    'Sample inputs for masked norm.'
    for ord in [2.0, 1, float('inf'), float('-inf'), 0]:
        for sample_input in sample_inputs_masked_reduction(op_info, device, dtype, requires_grad, **kwargs):
            (sample_input_args, sample_input_kwargs) = ((ord,) + sample_input.args, sample_input.kwargs.copy())
            yield SampleInput(sample_input.input.clone().requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)

def reference_masked_std_var(numpy_fn):
    if False:
        i = 10
        return i + 15
    ref = reference_reduction_numpy(numpy_fn)

    def func(input, dim=None, unbiased=None, *, correction=None, **kwargs):
        if False:
            return 10
        ddof = 1
        if unbiased is not None:
            ddof = 1 if unbiased else 0
        if correction is not None:
            ddof = correction
        if isinstance(dim, Sequence):
            dim = tuple(dim)
        return ref(input, dim, ddof=ddof, **kwargs)
    return func

def sample_inputs_masked_std_var(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    'Sample inputs for masked std/var.'
    kwargs['supports_multiple_dims'] = op_info.supports_multiple_dims
    from torch.testing._internal.common_methods_invocations import sample_inputs_std_var

    def masked_samples():
        if False:
            return 10
        for sample_input in sample_inputs_std_var(op_info, device, dtype, requires_grad, **kwargs):
            if len(sample_input.args) and isinstance(sample_input.args[0], bool):
                continue
            for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
                (sample_input_args, sample_input_kwargs) = (sample_input.args, dict(mask=mask, **sample_input.kwargs))
                yield SampleInput(sample_input.input.detach().requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)
                if not requires_grad and dtype.is_floating_point and (sample_input.input.ndim == 2) and (mask is not None) and (mask.shape == sample_input.input.shape):
                    for v in [torch.inf, -torch.inf, torch.nan]:
                        t = sample_input.input.detach()
                        t.diagonal(0, -2, -1).fill_(v)
                        yield SampleInput(t.requires_grad_(requires_grad), args=sample_input_args, kwargs=sample_input_kwargs)
    for sample_input in masked_samples():
        correction = sample_input.kwargs.get('correction')
        if correction is None:
            correction = int(sample_input.kwargs.get('unbiased', True))
        dim = sample_input.kwargs.get('dim', None)
        if sample_input.kwargs.get('mask') is None:
            orig_count = torch.masked.sum(torch.ones(sample_input.input.shape, dtype=torch.int64), dim, keepdim=True)
        else:
            inmask = torch.masked._input_mask(sample_input.input, *sample_input.args, **sample_input.kwargs)
            orig_count = torch.masked.sum(inmask.new_ones(sample_input.input.shape, dtype=torch.int64), dim, keepdim=True, mask=inmask)
        if orig_count.min() <= correction + 1:
            continue
        yield sample_input

def sample_inputs_masked_softmax(op_info, device, dtype, requires_grad, with_dtype=False, **kwargs):
    if False:
        i = 10
        return i + 15
    'Sample inputs for masked softmax, log_softmax, and softmin.\n\n    Masked normalization operator is a reduction operator with\n    trailing mask optional argument. A mask is a bool tensor with the\n    same shape as input or a shape that is broadcastable to input\n    shape.\n    '
    for sample_input in sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, with_dtype=with_dtype, **kwargs):
        for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
            yield SampleInput(sample_input.input.clone().requires_grad_(requires_grad), *sample_input.args, mask=mask, **sample_input.kwargs)

def sample_inputs_masked_cumops(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    'Sample inputs for masked cumsum and cumprod.'
    inputs: List[SampleInput] = []
    for sample_input in sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, **kwargs):
        for mask in _generate_masked_op_mask(sample_input.input.shape, device, **kwargs):
            if type(mask) != torch.Tensor:
                continue
            (sample_input_args, sample_input_kwargs) = (sample_input.args, dict(mask=mask, **sample_input.kwargs))
            if 'keepdim' in sample_input_kwargs:
                sample_input_kwargs.pop('keepdim')
            if sample_input_args:
                dim = sample_input.args[0]
            else:
                if 'dim' not in sample_input_kwargs:
                    continue
                dim = sample_input_kwargs.pop('dim')
                sample_input_args = (dim,)
            yield SampleInput(sample_input.input.clone().requires_grad_(requires_grad), *sample_input_args, **sample_input_kwargs)

def sample_inputs_masked_logaddexp(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    'Sample inputs for masked logaddexp.'
    shapes = [(S,), (S, S), (S, M, S)]
    input_mask_lists = [list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes]
    other_mask_lists = [list(_generate_masked_op_mask(shape, device, **kwargs)) for shape in shapes]
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for (shape, input_masks, other_masks) in zip(shapes, input_mask_lists, other_mask_lists):
        for (input_mask, other_mask) in zip(input_masks, other_masks):
            yield SampleInput(make_arg(shape), make_arg(shape), input_mask=input_mask, other_mask=other_mask)

def sample_inputs_masked_normalize(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    'Sample inputs for masked normalize.'
    for ord in [2.0, 1, float('inf'), float('-inf'), 0]:
        for sample_input in sample_inputs_softmax_variant(op_info, device, dtype, requires_grad, use_zero_dimensions=False, **kwargs):
            yield SampleInput(sample_input.input.clone().requires_grad_(requires_grad), ord, *sample_input.args, **sample_input.kwargs)
op_db: List[OpInfo] = [ReductionOpInfo('masked.sum', ref=reference_reduction_numpy(np.sum), method_variant=None, identity=0, nan_policy='propagate', supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, promotes_int_to_int64=True, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.skip('Failing on some jobs'), 'TestReductions', 'test_reference_masked', dtypes=(torch.bool, torch.int8, torch.int16, torch.int32)), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), decorators=[DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.001, rtol=0.05), torch.float16: tol(atol=0.001, rtol=0.005)}), 'TestReductions', 'test_reference_masked'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.001)}), 'TestReductions', 'test_ref_small_input'), DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.1, rtol=0.1), torch.float16: tol(atol=0.005, rtol=0.005)}), 'TestMasked', 'test_mask_layout')], sample_inputs_func=sample_inputs_masked_reduction, sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction, sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction), ReductionOpInfo('masked.prod', ref=prod_numpy, method_variant=None, identity=1, nan_policy='propagate', gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, promotes_int_to_int64=True, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Failing on some jobs'), 'TestReductions', 'test_reference_masked', dtypes=(torch.bool, torch.int8, torch.int16, torch.int32)), DecorateInfo('TestReductions', 'test_ref_small_input', dtypes=(torch.int8, torch.int16, torch.int32)), DecorateInfo(unittest.skip('Skipped!'), 'TestMasked', 'test_mask_layout', device_type='cuda', dtypes=(torch.bool, *integral_types(), *complex_types()))), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.01)}), 'TestReductions', 'test_reference_masked'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.001)}), 'TestReductions', 'test_ref_duplicate_values'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.001)}), 'TestReductions', 'test_ref_small_input'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.0015)}), 'TestMasked', 'test_mask_layout', device_type='cpu')], sample_inputs_func=sample_inputs_masked_reduction, sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction, sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction), OpInfo('masked.cumsum', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), method_variant=None, gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=sample_inputs_masked_cumops, gradcheck_wrapper=gradcheck_wrapper_masked_operation), OpInfo('masked.cumprod', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), method_variant=None, gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-05)}), 'TestCompositeCompliance', 'test_backward', device_type='cuda'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.002, rtol=0.002)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cuda')), sample_inputs_func=sample_inputs_masked_cumops, gradcheck_wrapper=gradcheck_wrapper_masked_operation), ReductionOpInfo('masked.amax', nan_policy='propagate', supports_out=False, dtypes=all_types_and(torch.float16, torch.bfloat16), supports_sparse=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse_csr=True, ref=reference_reduction_numpy(np.amax), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMasked', 'test_mask_layout', dtypes=(torch.bool, *integral_types(), *complex_types()))), sample_inputs_func=sample_inputs_masked_reduction, sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction, sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction, gradcheck_wrapper=gradcheck_wrapper_masked_operation), ReductionOpInfo('masked.amin', nan_policy='propagate', supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=all_types_and(torch.float16, torch.bfloat16), supports_sparse=True, supports_sparse_csr=True, ref=reference_reduction_numpy(np.amin), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMasked', 'test_mask_layout', dtypes=(torch.bool, *integral_types(), *complex_types()))), sample_inputs_func=sample_inputs_masked_reduction, sample_inputs_sparse_coo_func=sample_inputs_sparse_coo_masked_reduction, sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction, gradcheck_wrapper=gradcheck_wrapper_masked_operation), ReductionOpInfo('masked.argmax', supports_out=False, supports_multiple_dims=False, supports_autograd=False, dtypes=all_types_and(torch.float16, torch.bfloat16), ref=reference_reduction_numpy(np.argmax, supports_keepdims=False), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_reference_masked'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=sample_inputs_masked_reduction, gradcheck_wrapper=gradcheck_wrapper_masked_operation), ReductionOpInfo('masked.argmin', supports_out=False, supports_multiple_dims=False, supports_autograd=False, dtypes=all_types_and(torch.float16, torch.bfloat16), ref=reference_reduction_numpy(np.argmin, supports_keepdims=False), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_reference_masked'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=sample_inputs_masked_reduction, gradcheck_wrapper=gradcheck_wrapper_masked_operation), ReductionOpInfo('masked.mean', ref=reference_reduction_numpy(np.mean) if np.lib.NumpyVersion(np.__version__) >= '1.20.2' else None, method_variant=None, nan_policy='propagate', supports_out=False, supports_sparse_csr=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_ref_duplicate_values', dtypes=(torch.bool,)), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_reference_masked', dtypes=(torch.bool,)), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_ref_small_input', dtypes=(torch.bool,)), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMasked', 'test_mask_layout', dtypes=(torch.bool, *integral_types(), *complex_types()))), decorators=[DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.001, rtol=0.05), torch.float16: tol(atol=0.001, rtol=0.001)}), 'TestReductions', 'test_reference_masked'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.001)}), 'TestReductions', 'test_ref_small_input'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.002)}), 'TestSparseCompressed', 'test_consistency', device_type='cuda')], sample_inputs_func=sample_inputs_masked_reduction, sample_inputs_sparse_csr_func=sample_inputs_sparse_csr_masked_reduction, gradcheck_wrapper=gradcheck_wrapper_masked_operation), OpInfo('masked.median', dtypes=floating_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.float16), method_variant=None, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=partial(sample_inputs_masked_softmax, use_zero_dimensions=False), gradcheck_wrapper=gradcheck_wrapper_masked_operation), ReductionOpInfo('masked.norm', identity=0, method_variant=None, nan_policy='propagate', supports_out=False, promotes_int_to_float=True, dtypes=floating_types_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_masked_norm, gradcheck_wrapper=gradcheck_wrapper_masked_operation), ReductionOpInfo('masked.var', ref=reference_masked_std_var(np.var) if np.lib.NumpyVersion(np.__version__) >= '1.20.2' else None, method_variant=None, nan_policy='propagate', supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, promotes_int_to_float=True, dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01), torch.bfloat16: tol(atol=0.001, rtol=0.001)}), 'TestReductions', 'test_reference_masked'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestReductions', 'test_ref_small_input'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestMasked', 'test_reference_masked'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01), torch.bfloat16: tol(atol=0.001, rtol=0.001)}), 'TestMasked', 'test_reference_masked')], sample_inputs_func=sample_inputs_masked_std_var, gradcheck_wrapper=gradcheck_wrapper_masked_operation, check_batched_grad=True), ReductionOpInfo('masked.std', ref=reference_masked_std_var(np.std) if np.lib.NumpyVersion(np.__version__) >= '1.20.2' else None, method_variant=None, nan_policy='propagate', gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, promotes_int_to_float=True, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), decorators=[DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.01, rtol=0.01), torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestReductions', 'test_reference_masked'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestReductions', 'test_ref_small_input'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01), torch.bfloat16: tol(atol=0.005, rtol=0.0005)}), 'TestMasked', 'test_reference_masked')], sample_inputs_func=sample_inputs_masked_std_var, gradcheck_wrapper=gradcheck_wrapper_masked_operation, check_batched_grad=True), OpInfo('masked.softmax', method_variant=None, dtypes=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_masked_softmax, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), gradcheck_wrapper=gradcheck_wrapper_masked_operation, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('masked.log_softmax', method_variant=None, dtypes=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_masked_softmax, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), decorators=[DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.01, rtol=0.01)}), 'TestMasked', 'test_reference_masked')], gradcheck_wrapper=gradcheck_wrapper_masked_operation, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('masked.softmin', method_variant=None, dtypes=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_masked_softmax, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), gradcheck_wrapper=gradcheck_wrapper_masked_operation, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('masked.normalize', method_variant=None, dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_masked_normalize, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), gradcheck_wrapper=gradcheck_wrapper_masked_operation, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('masked.logaddexp', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_gradgrad')), sample_inputs_func=sample_inputs_masked_logaddexp, gradcheck_wrapper=gradcheck_wrapper_masked_pointwise_operation), ReductionOpInfo('masked.logsumexp', dtypes=all_types_and(torch.half, torch.bfloat16), method_variant=None, nan_policy='propagate', supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_empty_tensor_empty_slice'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestDecomp', 'test_comprehensive')), sample_inputs_func=sample_inputs_masked_reduction, gradcheck_wrapper=gradcheck_wrapper_masked_operation)]