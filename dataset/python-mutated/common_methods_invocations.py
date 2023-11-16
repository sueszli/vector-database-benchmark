from functools import wraps, partial
from itertools import product, chain, islice
import itertools
import functools
import copy
import operator
import random
import unittest
import math
import enum
import torch
import numpy as np
from torch import inf, nan
from typing import Any, Dict, List, Tuple, Union, Sequence
from torch.testing import make_tensor
from torch.testing._internal.common_dtype import _dispatch_dtypes, floating_types, floating_types_and, complex_types, floating_and_complex_types, floating_and_complex_types_and, all_types_and_complex_and, all_types_and, all_types_and_complex, integral_types_and, all_types, empty_types, complex_types_and, integral_types, custom_types
from torch.testing._internal.common_device_type import onlyCPU, onlyCUDA, onlyNativeDeviceTypes, disablecuDNN, skipCUDAIfNoMagma, skipCUDAIfNoMagmaAndNoCusolver, skipCUDAIfNoCusolver, skipCPUIfNoLapack, skipCPUIfNoFFT, skipCUDAIf, precisionOverride, skipCPUIfNoMklSparse, toleranceOverride, tol
from torch.testing._internal.common_cuda import SM53OrLater, SM60OrLater, SM80OrLater, SM90OrLater, with_tf32_off, TEST_CUDNN, _get_torch_cuda_version, _get_torch_rocm_version
from torch.testing._internal.common_utils import make_fullrank_matrices_with_distinct_singular_values, TEST_WITH_ROCM, IS_WINDOWS, IS_MACOS, TEST_SCIPY, torch_to_numpy_dtype_dict, TEST_WITH_ASAN, GRADCHECK_NONDET_TOL, freeze_rng_state, slowTest, TEST_WITH_SLOW, TEST_WITH_TORCHINDUCTOR
import torch._refs as refs
import torch._refs.nn.functional
import torch._refs.special
import torch._refs.linalg
import torch._prims as prims
from torch.utils import _pytree as pytree
from packaging import version
from torch.testing._internal.opinfo.core import L, M, S, XS, _NOTHING, _getattr_qual, DecorateInfo, SampleInput, ErrorInput, AliasInfo, NumericsFilter, OpInfo, _generate_reduction_inputs, _generate_reduction_kwargs, sample_inputs_reduction, ReductionOpInfo, reference_inputs_elementwise_binary, make_error_inputs_elementwise_binary, generate_elementwise_binary_tensors, generate_elementwise_binary_arbitrarily_strided_tensors, generate_elementwise_binary_small_value_tensors, generate_elementwise_binary_large_value_tensors, generate_elementwise_binary_extremal_value_tensors, generate_elementwise_binary_broadcasting_tensors, generate_elementwise_binary_with_scalar_samples, generate_elementwise_binary_with_scalar_and_type_promotion_samples, generate_elementwise_binary_noncontiguous_tensors, sample_inputs_elementwise_binary, BinaryUfuncInfo, sample_inputs_elementwise_unary, generate_elementwise_unary_tensors, generate_elementwise_unary_small_value_tensors, generate_elementwise_unary_large_value_tensors, generate_elementwise_unary_extremal_value_tensors, reference_inputs_elementwise_unary, UnaryUfuncInfo, sample_inputs_spectral_ops, SpectralFuncType, SpectralFuncInfo, ShapeFuncInfo, sample_inputs_foreach, ForeachFuncInfo, gradcheck_wrapper_hermitian_input, gradcheck_wrapper_triangular_input, gradcheck_wrapper_triangular_input_real_positive_diagonal, gradcheck_wrapper_masked_operation, gradcheck_wrapper_masked_pointwise_operation, clone_sample
from torch.testing._internal.opinfo.refs import _find_referenced_opinfo, _inherit_constructor_args, PythonRefInfo, ReductionPythonRefInfo, ElementwiseUnaryPythonRefInfo, ElementwiseBinaryPythonRefInfo
from torch.testing._internal.opinfo.utils import np_unary_ufunc_integer_promotion_wrapper, reference_reduction_numpy, prod_numpy
from torch.testing._internal import opinfo
from torch.testing._internal.opinfo.definitions.linalg import sample_inputs_linalg_cholesky, sample_inputs_linalg_cholesky_inverse, sample_inputs_cross, sample_inputs_linalg_qr_geqrf, sample_inputs_linalg_invertible, sample_inputs_lu_solve, sample_inputs_legacy_solve, sample_inputs_svd, sample_inputs_linalg_det_logdet_slogdet, sample_inputs_linalg_lu, sample_inputs_diagonal_diag_embed, error_inputs_diagonal_diag_embed
from torch.testing._internal.opinfo.definitions.special import sample_inputs_i0_i1, sample_inputs_polygamma, reference_polygamma
from torch.testing._internal.opinfo.definitions._masked import sample_inputs_softmax_variant
from torch.testing._internal.opinfo.definitions.sparse import error_inputs_sparse_like_fns, sample_inputs_sparse_like_fns, error_inputs_sparse_mul, sample_inputs_sparse_mul, error_inputs_sparse_reduction_sum, sample_inputs_sparse_reduction_sum
if TEST_SCIPY:
    from scipy import stats
    import scipy.spatial
    import scipy.special

def close_to_int(x, eps=0.1):
    if False:
        return 10
    if x.is_complex():
        y = torch.abs(torch.view_as_complex(torch.frac(torch.view_as_real(x))))
    else:
        y = torch.abs(torch.frac(x))
    return (y < eps) | (y > 1 - eps)

def sample_inputs_slice(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_input = partial(make_tensor, device=device, dtype=dtype, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_input(3), 0)
    yield SampleInput(make_input(20, 30, 40), dim=1, start=1, end=-2)
    yield SampleInput(make_input(20, 30, 40), dim=1, start=1, end=-2, step=3)
    yield SampleInput(make_input(20, 30, 40), dim=0, start=-10, end=-2, step=2)

def sample_inputs_tensor_split(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_input = partial(make_tensor, device=device, dtype=dtype, low=None, high=None, requires_grad=requires_grad)
    args_cases = ((torch.tensor([1, 2, 3]),), (torch.tensor(1),), (torch.tensor([1, 2, 3]), 1), (torch.tensor([1, 4, 2, 5, 3, 6])[::2], 1), ((2, 4),), ((2, 4), 1), ((2, 4), -1), (3,), (3, 1), (3, -1))
    for args in args_cases:
        yield SampleInput(make_input((S, S, S)), args=args)

def sample_inputs_hsplit(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(6), 2)
    yield SampleInput(make_arg(S, S, S), [1, 2, 3])

def sample_inputs_vsplit(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(6, S), 2)
    yield SampleInput(make_arg(S, S, S), [1, 2, 3])

def sample_inputs_dsplit(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(S, S, S), [1, 2, 3])
    yield SampleInput(make_arg(S, S, 6), 2)

def error_inputs_hsplit(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    err_msg1 = 'torch.hsplit requires a tensor with at least 1 dimension, but got a tensor with 0 dimensions!'
    yield ErrorInput(SampleInput(make_arg(()), 0), error_regex=err_msg1)
    err_msg2 = f'torch.hsplit attempted to split along dimension 1, but the size of the dimension {S} is not divisible by the split_size 0!'
    yield ErrorInput(SampleInput(make_arg((S, S, S)), 0), error_regex=err_msg2)
    err_msg3 = 'received an invalid combination of arguments.'
    yield ErrorInput(SampleInput(make_arg((S, S, S)), 'abc'), error_type=TypeError, error_regex=err_msg3)

def error_inputs_vsplit(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    err_msg1 = 'torch.vsplit requires a tensor with at least 2 dimension, but got a tensor with 1 dimensions!'
    yield ErrorInput(SampleInput(make_arg(S), 0), error_regex=err_msg1)
    err_msg2 = f'torch.vsplit attempted to split along dimension 0, but the size of the dimension {S} is not divisible by the split_size 0!'
    yield ErrorInput(SampleInput(make_arg(S, S, S), 0), error_regex=err_msg2)
    err_msg3 = 'received an invalid combination of arguments.'
    yield ErrorInput(SampleInput(make_arg(S, S, S), 'abc'), error_type=TypeError, error_regex=err_msg3)

def error_inputs_dsplit(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    err_msg1 = 'torch.dsplit requires a tensor with at least 3 dimension, but got a tensor with 1 dimensions!'
    yield ErrorInput(SampleInput(make_arg(S), 0), error_regex=err_msg1)
    err_msg2 = f'torch.dsplit attempted to split along dimension 2, but the size of the dimension {S} is not divisible by the split_size 0!'
    yield ErrorInput(SampleInput(make_arg(S, S, S), 0), error_regex=err_msg2)

def sample_inputs_as_strided(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = (((1,), (1,), (1,), 0), ((3, 3), (2, 2), (1, 2), 0), ((3, 3), (2, 2), (1, 2), 1), ((16,), (2, 2, 2, 2), (1, 1, 1, 1), 0), ((16,), (2, 1, 1, 2), (1, 7, 7, 1), 0))
    for (input_shape, output_shape, stride, storage_offset) in test_cases:
        input_t = make_arg(input_shape)
        kwargs = dict(storage_offset=storage_offset)
        yield SampleInput(input_t, args=(output_shape, stride), kwargs=kwargs)

def sample_inputs_as_strided_partial_views(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10

    def make_arg():
        if False:
            for i in range(10):
                print('nop')
        base = make_tensor((20,), device=device, dtype=dtype)
        return base[5:15].requires_grad_(requires_grad)
    yield SampleInput(make_arg(), (2, 2), (1, 2))
    yield SampleInput(make_arg(), (2, 2), (1, 2), storage_offset=0)
    yield SampleInput(make_arg(), (2, 2), (1, 2), storage_offset=10)

def sample_inputs_as_strided_scatter(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = [((1,), (), (), 0), ((1,), (1,), (1,), 0), ((3, 3), (2, 2), (1, 2), 0), ((3, 3), (2, 2), (1, 2), 1), ((3, 3), (2, 2), (2, 1), 0), ((16,), (2, 2, 2, 2), (8, 4, 2, 1), 0), ((16,), (2, 1, 1, 2), (1, 2, 4, 8), 0)]
    for (input_shape, output_shape, stride, storage_offset) in test_cases:
        input_t = make_arg(input_shape)
        input_src = make_arg(output_shape)
        yield SampleInput(input_t, input_src, output_shape, stride, storage_offset=storage_offset)

def error_inputs_as_strided_scatter(op_info, device, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=torch.float32, requires_grad=False)
    input_t = make_arg([4, 4])
    input_src = make_arg([2, 2])
    yield ErrorInput(SampleInput(input_t, input_src, [2, 2], [200, 200], storage_offset=0), error_regex='itemsize 4 requiring a storage size of 1604 are out of bounds for storage of size 64')

def sample_inputs_combinations(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    inputs = ((0,), (0, 1), (0, 1, 2, 3))
    rvals = [1, 2, 4]
    products = product(inputs, rvals, [False, True])
    for (input_data, r, with_replacement) in products:
        input_t = torch.tensor(input_data, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(input_t, r=r, with_replacement=with_replacement)

def sample_inputs_cartesian_prod(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(torch.tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    a = make_arg((0,))
    b = make_arg((0, 1))
    c = make_arg((0, 1, 2, 3))
    yield SampleInput(a)
    yield SampleInput(a, b)
    yield SampleInput(a, b, c)

def sample_inputs_cosine_similarity(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[tuple, dict] = (((S, S), {'dim': 1}), ((S, 2), {'dim': -1}), ((S,), {'dim': 0, 'eps': 0.5}), ((), {'dim': 0}), ((S, S, M), {'dim': 2}), ((S, S), {}))
    for (input_shape, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(input_shape),), kwargs=kwargs)
    yield SampleInput(make_arg((1, 2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -1})
    yield SampleInput(make_arg((1, 2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -2})
    yield SampleInput(make_arg((2, 3)), args=(make_arg((2, 1, 3)),), kwargs={'dim': -1})

def sample_inputs_item(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    cases = ((), (), 1, (1,))
    for shape in cases:
        yield SampleInput(make_arg(shape))

def error_inputs_item(op, device, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=torch.float32, device=device, requires_grad=False)
    cases = (M, (S,), (S, S), (S, M, L))
    for shape in cases:
        yield ErrorInput(SampleInput(make_arg(shape)), error_type=RuntimeError, error_regex='elements cannot be converted to Scalar')

def sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    cases: Tuple[Tuple[int], dict] = (((S, S, S), {'training': True, 'momentum': 0.5, 'eps': 0.6}), ((3, 2, 4), {'training': False, 'momentum': -1.2}), ((3, 1), {'training': True, 'momentum': 0.0}), ((0,), {'training': True}), ((0,), {'training': False}), ((3, 2, 3, 4), {'training': True, 'momentum': -1.0, 'eps': 0.5}), ((3, 2, 3, 4), {'training': False, 'momentum': -1.0, 'eps': 0.5}), ((2, 1), {}))
    for (input_shape, kwargs) in cases:
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight = make_arg(channels) if channels > 0 else None
        bias = make_arg(channels) if channels > 0 else None
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        yield SampleInput(make_arg(input_shape), args=(running_mean, running_var, weight, bias), kwargs=kwargs)
    weights = [channels, None, None]
    biases = [None, channels, None]
    is_training = [True, False, False]
    for (weight, bias, training) in zip(weights, biases, is_training):
        yield SampleInput(make_arg(input_shape), args=(running_mean, running_var, make_arg(channels), make_arg(channels)), kwargs={'training': training})
    yield SampleInput(make_arg((1, 2, 3)), args=(None, None, None, None), kwargs={'training': True})

def sample_inputs_softmax_backward_data(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = [((S,), 0), ((S, S), 0), ((S, M, S), -1)]
    input_dtypes = [dtype]
    if dtype == torch.float and device == 'cuda':
        input_dtypes += [torch.float16]
    for ((shape, dim), input_dtype) in product(cases, input_dtypes):
        input = make_arg(shape)
        output = torch.nn.functional.softmax(input, dim=dim, dtype=input_dtype)
        yield SampleInput(make_arg(shape), output, dim, input_dtype)

def sample_inputs_native_batch_norm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    samples = sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs)
    for sample in samples:
        if sample.input.numel() == 0:
            continue
        args = sample.args
        training = sample.kwargs.get('training', True)
        momentum = sample.kwargs.get('momentum', 0.5)
        eps = sample.kwargs.get('eps', 1e-05)
        yield SampleInput(sample.input, args=(args[2], args[3], args[0], args[1], training, momentum, eps))

def sample_inputs__native_batch_norm_legit(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    samples = sample_inputs_batch_norm(op_info, device, dtype, requires_grad, **kwargs)
    for sample in samples:
        if sample.input.numel() == 0:
            continue
        args = sample.args
        training = sample.kwargs.get('training', True)
        momentum = sample.kwargs.get('momentum', 0.5)
        eps = sample.kwargs.get('eps', 1e-05)
        if args[0] is not None and args[1] is not None:
            yield SampleInput(sample.input, args=(args[2], args[3], args[0], args[1], training, momentum, eps))
        else:
            yield SampleInput(sample.input, args=(args[2], args[3], training, momentum, eps))

def sample_inputs_nn_activation_relu(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = ((), (S,), (S, S), (S, M, S))
    for shape in cases:
        yield SampleInput(make_arg(shape))

def sample_inputs_prelu(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    op_kwargs = op_info.sample_kwargs(device, dtype, None)[0]
    yield from sample_inputs_elementwise_unary(op_info, device, dtype, requires_grad, op_kwargs=op_kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = ((), (S,), (S, S), (S, M, S))
    for shape in cases:
        for weight in [-1.0, 0.0, 0.8, 1.0]:
            weight_tensor = torch.tensor(weight, device=device, dtype=dtype, requires_grad=requires_grad)
            yield SampleInput(make_arg(shape), args=(weight_tensor,))
        channel_size = shape[1] if len(shape) >= 2 else 1
        yield SampleInput(make_arg(shape), args=(make_arg((channel_size,)),))
    weight_tensor = torch.tensor(1.0, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)), kwargs=dict(weight=weight_tensor))
    yield SampleInput(make_arg((S, S)), kwargs=dict(weight=make_arg((S,))))

def reference_inputs_prelu(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_prelu(op, device, dtype, requires_grad, **kwargs)
    yield from reference_inputs_elementwise_unary(op, device, dtype, requires_grad, **kwargs)

def sample_kwargs_prelu_scalar_weight(device, dtype, input):
    if False:
        return 10
    weight = torch.rand(tuple(), device=device, dtype=dtype)
    if dtype == torch.bfloat16:
        weight_cpu = weight.to(dtype=torch.float32, device='cpu')
    else:
        weight_cpu = weight.cpu()
    np_weight = weight_cpu.numpy()
    return ({'weight': weight}, {'weight': np_weight})

def error_inputs_prelu(op, device):
    if False:
        i = 10
        return i + 15
    inp = make_tensor(tuple(), device=device, dtype=torch.float32)
    weight = make_tensor((2,), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(inp, kwargs={'weight': weight}), error_regex='Not allow zero-dim input tensor.')
    inp = make_tensor((2, 8, 3), device=device, dtype=torch.float32)
    weight = make_tensor((9,), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(inp, kwargs={'weight': weight}), error_regex='Mismatch of parameter numbers and input channel size.')
    inp = make_tensor((2, 8, 3), device=device, dtype=torch.float32)
    weight = make_tensor((2, 4), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(inp, kwargs={'weight': weight}), error_regex='prelu: Expected `weight` to be a scalar or 1D tensor, but got: ndim = 2')

def sample_inputs_norm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = [((S, S), (2,), '2'), ((S, S), (0,), '0'), ((S, S), (0.5,), '0_5'), ((S, S), (1,), '1'), ((S, S), (3,), '3'), ((S, S), (-1,), 'neg_1'), ((S, S), (-2,), 'neg_2'), ((S, S), (-0.5,), 'neg_0_5'), ((S, S), (-1.5,), 'neg_1_5')]
    cases_nonzero_input = (((S, S, S), (1.5,), '1_5_default'), ((S, S, S), (1.5, 1), '1_5_dim'), ((S, S, S), (1.5, -1), '1_5_neg_dim'), ((S, S, S), (1.5, 1, True), 'keepdim_1_5_dim'), ((S, S, S), (1.5, -1, True), 'keepdim_1_5_neg_dim'))
    cases_posdim = (((S, S), (-2, 1), 'neg_2_dim'), ((S, S), (-1, 1), 'neg_1_dim'), ((S, S), (0, 1), '0_dim'), ((S, S), (1, 1), '1_dim'), ((S, S), (2, 1), '2_dim'), ((S, S), (3, 1), '3_dim'), ((S, S, S), (2, 1), '2_dim'), ((S, S, S), (3, 1), '3_dim'), ((S, S, S), (2, 1, True), 'keepdim_2_dim'), ((S, S, S), (3, 1, True), 'keepdim_3_dim'), ((), (2, 0), '2_dim_scalar'), ((), (3, 0), '3_dim_scalar'), ((), (2, 0, True), 'keepdim_2_dim_scalar'), ((), (3, 0, True), 'keepdim_3_dim_scalar'))
    cases_negdim = ((shape, args[:1] + (-args[1],) + args[2:], name.replace('_dim', '_neg_dim')) for (shape, args, name) in cases_posdim)
    for (shape, args, name) in itertools.chain(cases, cases_posdim, cases_negdim):
        yield SampleInput(make_arg(shape), args=args, name=name)
    for (shape, args, name) in cases_nonzero_input:
        yield SampleInput(make_arg(shape, exclude_zero=True), args=args, name=name)

def sample_inputs_norm_fro(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((S, S), (), 'default'), ((S, S), ('fro',), 'fro_default'), ((S, S), ('fro', [0, 1]), 'fro'))
    for (shape, args, name) in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)

def sample_inputs_norm_nuc(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((S, S), ('nuc',), 'nuc'), ((S, S, S), ('nuc', [1, 2]), 'nuc_batched'))
    for (shape, args, name) in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)

def sample_inputs_norm_inf(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((S, S), (-inf,), '-inf'), ((S, S), (inf,), 'inf'), ((S, S), (inf, 1), 'inf_2_dim'), ((S, S), (inf, -1), 'inf_2_neg_dim'))
    for (shape, args, name) in cases:
        yield SampleInput(make_arg(shape), args=args, name=name)

def sample_inputs_equal(op, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes = (((), ()), ((S,), ()), ((), (S,)), ((S, 1), (S,)), ((M, S), ()), ((S, S), (S, S)))
    for (shape_lhs, shape_rhs) in shapes:
        lhs = make_arg(shape_lhs)
        rhs = make_arg(shape_rhs)
        broadcasts_input = shape_lhs != torch.broadcast_shapes(shape_lhs, shape_rhs)
        yield SampleInput(lhs, args=(rhs,), broadcasts_input=broadcasts_input)
        if shape_lhs == shape_rhs:
            yield SampleInput(lhs, args=(lhs.clone().detach_(),))

def sample_inputs_jiterator(op, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes = (((), ()), ((S,), ()), ((S, 1), (S,)), ((M, S), ()), ((S, M, S), (M, S)), ((S, M, S), (S, M, S)), ((M, 1, S), (M, S)), ((M, 1, S), (1, M, S)), ((0, 1, 3), (0, 10, 3)))
    num_inputs = kwargs.get('num_inputs')
    sample_kwargs = kwargs.get('sample_kwargs', {})
    for (shape_lhs, shape_rhs) in shapes:
        lhs = make_arg(shape_lhs)
        args = []
        for i in range(num_inputs - 1):
            args.append(make_arg(shape_rhs))
        broadcasts_input = shape_lhs != torch.broadcast_shapes(shape_lhs, shape_rhs)
        yield SampleInput(lhs, args=tuple(args), kwargs=sample_kwargs, broadcasts_input=broadcasts_input)

def sample_inputs_broadcast_shapes(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    shapes = (((), ()), ((S,), ()), ((S, 1), (S,)), ((S, 1), S), ((M, S), ()), ((S, M, S), (M, S)), ((S, M, S), (S, M, S)), ((M, 1, S), (M, S)), ((M, 1, S), (1, M, S)), ((0, 1, 3), (0, 10, 3)))
    for shape in shapes:
        (inp, *arg0) = shape
        yield SampleInput(inp, args=tuple(arg0))

def sample_inputs_add_sub(op, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    yield from sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)
    if dtype is not torch.bool:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': 2})
    else:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': True})
    neg_alpha = -3.125 if dtype.is_floating_point or dtype.is_complex else -3
    lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
    rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)
    if dtype is not torch.bool:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': neg_alpha})
    else:
        yield SampleInput(lhs, args=(rhs,), kwargs={'alpha': False})

def error_inputs_arange(op, device, **kwargs):
    if False:
        return 10
    yield ErrorInput(SampleInput(0, args=(3, 0)), error_type=RuntimeError, error_regex='step must be nonzer')
    yield ErrorInput(SampleInput(0, args=(-3, 2)), error_type=RuntimeError, error_regex='bound inconsistent with step sign')
    yield ErrorInput(SampleInput(0, args=(3, -2)), error_type=RuntimeError, error_regex='bound inconsistent with step sign')
    yield ErrorInput(SampleInput(0, args=(float('inf'), 2)), error_type=RuntimeError, error_regex='unsupported range')
    yield ErrorInput(SampleInput(float('-inf'), args=(1, 2)), error_type=RuntimeError, error_regex='unsupported range')

def sample_inputs_arange(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    int_samples = ((-1, 2, 2), (2, -3, -1), (1, 1, 1), (1, 1, -1), (0, -8, -4), (1, 5, 2), (False, True, True), (0, 1, None), (None, 3, None))

    def to_float(start, end, step):
        if False:
            for i in range(10):
                print('nop')
        start = start + 0.1 if start is not None else None
        end = end + 0.1
        step = float(step) if step is not None else None
        return (start, end, step)
    float_samples = ((0.0, -8.0 - 1e-06, -4.0), (1.0, 5.0 + 1e-06, 2.0), (0.0, -8.0, -4.0), (1.0, 5.0, 2.0), *(to_float(start, end, step) for (start, end, step) in int_samples))
    large_samples = ((0, 10000, None),)
    samples = int_samples + float_samples
    if dtype not in (torch.int8, torch.uint8):
        samples += large_samples
    for (start, end, step) in samples:
        if start is None:
            assert step is None
            yield SampleInput(end, kwargs={'dtype': dtype, 'device': device})
            yield SampleInput(0, kwargs={'end': end, 'dtype': dtype, 'device': device})
        elif step is None:
            yield SampleInput(start, args=(end,), kwargs={'dtype': dtype, 'device': device})
        else:
            yield SampleInput(start, args=(end, step), kwargs={'dtype': dtype, 'device': device})
    yield SampleInput(2)
    yield SampleInput(1, args=(3, 1))

def sample_inputs_randn(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    shapes = ((M,), (S, S))
    for shape in shapes:
        yield SampleInput(input=shape, kwargs=dict(dtype=dtype, device=device, requires_grad=requires_grad))

def sample_inputs_normal(op, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (((S, S), 0, 5), ((S, S, S), -2, 0.5))
    for (shape, mean, std) in samples:
        yield SampleInput(make_arg(shape), args=(mean, std))

def error_inputs_normal(op, device, **kwargs):
    if False:
        print('Hello World!')
    t = torch.zeros([10], device=device)
    invalid_std = -1
    yield ErrorInput(SampleInput(t, args=(0, invalid_std)), error_type=RuntimeError, error_regex=f'normal expects std >= 0.0, but found std {invalid_std}')

def sample_inputs_cauchy(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (((M,), 0, 0.5), ((S, S), 0, 1), ((S, S, S), -2, 1))
    for (shape, median, gamma) in samples:
        yield SampleInput(make_arg(shape), args=(median, gamma))

def error_inputs_cauchy(op, device, **kwargs):
    if False:
        i = 10
        return i + 15
    t = torch.zeros([10], device=device)
    invalid_scale = 0
    yield ErrorInput(SampleInput(t, args=(0, invalid_scale)), error_type=RuntimeError, error_regex=f'cauchy_ expects sigma > 0.0, but found sigma={invalid_scale}')

def sample_inputs_exponential(op, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (((M,), 0.5), ((S, S), 1), ((S, S, S), 1.5))
    for (shape, rate) in samples:
        yield SampleInput(make_arg(shape), args=(rate,))

def error_inputs_exponential(op, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    t = torch.zeros([10], device=device)
    invalid_rate = 0
    yield ErrorInput(SampleInput(t, args=(invalid_rate,)), error_type=RuntimeError, error_regex=f'exponential_ expects lambda > 0.0, but found lambda={invalid_rate}')

def sample_inputs_geometric(op, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (((M,), 0.2), ((S, S), 0.5), ((S, S, S), 0.8))
    for (shape, rate) in samples:
        yield SampleInput(make_arg(shape), args=(rate,))

def error_inputs_geometric(op, device, **kwargs):
    if False:
        print('Hello World!')
    t = torch.zeros([10], device=device)
    neg_prob = -1
    yield ErrorInput(SampleInput(t, args=(neg_prob,)), error_type=RuntimeError, error_regex=f'geometric_ expects p to be in \\(0, 1\\), but got p={neg_prob}')

def sample_inputs_log_normal(op, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (((M,), 0, 0.25), ((S, S), 0.5, 1), ((S, S, S), 0, 0.5))
    for (shape, mean, std) in samples:
        yield SampleInput(make_arg(shape), args=(mean, std))

def error_inputs_log_normal(op, device, **kwargs):
    if False:
        print('Hello World!')
    t = torch.zeros([10], device=device)
    invalid_std = 0
    yield ErrorInput(SampleInput(t, args=(0, invalid_std)), error_type=RuntimeError, error_regex=f'log_normal_ expects std > 0.0, but found std={invalid_std}')

def sample_inputs_uniform(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=False)
    samples = (((M,), -100, 100), ((S, S), 0, 1), ((S, S, S), 1, 2))
    for (shape, hi, lo) in samples:
        yield SampleInput(make_arg(shape), args=(hi, lo))

def sample_inputs_ones_zeros(op, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    sizes = ((M,), (S, S))
    for size in sizes:
        yield SampleInput(size, kwargs={'dtype': dtype, 'device': device})

def sample_inputs_full(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10

    def get_val(dtype):
        if False:
            while True:
                i = 10
        return make_tensor([], dtype=dtype, device='cpu').item()
    sizes = ((M,), (S, S))
    fill_values = [get_val(dtype), get_val(torch.int)]
    for (size, fill_value) in product(sizes, fill_values):
        yield SampleInput(size, fill_value, dtype=dtype, device=device)

def error_inputs_uniform(op, device, **kwargs):
    if False:
        while True:
            i = 10
    t = torch.zeros([10], device=device)
    yield ErrorInput(SampleInput(t, args=(3, -1)), error_type=RuntimeError, error_regex='uniform_ expects to return a \\[from, to\\) range, but found from=3 > to=-1')

def error_inputs_linspace(op, device, **kwargs):
    if False:
        return 10
    yield ErrorInput(SampleInput(0, args=(3, -1)), error_type=RuntimeError, error_regex='number of steps must be non-negative')
    yield ErrorInput(SampleInput(0, args=(3, 1.0)), error_type=TypeError, error_regex='received an invalid combination of arguments - got \\(int, int, float')
    yield ErrorInput(SampleInput(torch.tensor([1, 1], device=device), args=(torch.tensor([3, 3], device=device), 1)), error_type=RuntimeError, error_regex='only supports 0-dimensional start and end tensors')

def sample_inputs_linspace(op, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ends = (-3, 0, 1, 4, 50)
    starts = (-2.0, 0, 4.3, 50)
    nsteps = (0, 1, 50)
    cases = list(product(starts, ends, nsteps)) + [(0, 7, 50)]
    for (start, end, nstep) in cases:
        if dtype == torch.uint8 and (end < 0 or start < 0):
            continue
        yield SampleInput(start, args=(end, nstep), kwargs={'dtype': dtype, 'device': device})
    yield SampleInput(1, args=(3, 1))

def sample_inputs_linspace_tensor_overload(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    ends = (-3, 0, 1, 4, 50)
    starts = (-2.0, 0, 4.3, 50)
    nsteps = (0, 1, 50)
    is_start_end_tensors = ((True, True), (True, False), (False, True))
    make_arg = partial(torch.tensor, device=device, requires_grad=False)
    cases = list(product(starts, ends, nsteps, is_start_end_tensors)) + [(0, 7, 50, (True, True))]
    for (start, end, nstep, (is_start_tensor, is_end_tensor)) in cases:
        if dtype == torch.uint8 and (end < 0 or start < 0):
            continue
        tensor_options = {'dtype': dtype, 'device': device}
        if is_start_tensor:
            start = make_arg(start, dtype=torch.float32 if isinstance(start, float) else torch.int64)
        if is_end_tensor:
            end = make_arg(end, dtype=torch.float32 if isinstance(end, float) else torch.int64)
        yield SampleInput(start, args=(end, nstep), kwargs=tensor_options)
    yield SampleInput(1, args=(3, 1))

def sample_inputs_logspace(op, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ends = (-3, 0, 1.2, 2, 4)
    starts = (-2.0, 0, 1, 2, 4.3)
    nsteps = (0, 1, 2, 4)
    bases = (2.0, 1.1) if dtype in (torch.int8, torch.uint8) else (None, 2.0, 3.0, 1.1, 5.0)
    for (start, end, nstep, base) in product(starts, ends, nsteps, bases):
        if dtype == torch.uint8 and end < 0 or start < 0:
            continue
        if nstep == 1 and isinstance(start, float) and (not (dtype.is_complex or dtype.is_floating_point)):
            continue
        if base is None:
            yield SampleInput(start, args=(end, nstep), kwargs={'dtype': dtype, 'device': device})
        else:
            yield SampleInput(start, args=(end, nstep, base), kwargs={'dtype': dtype, 'device': device})
    yield SampleInput(1, args=(3, 1, 2.0))

def sample_inputs_logspace_tensor_overload(op, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    ends = (-3, 0, 1.2, 2, 4)
    starts = (-2.0, 0, 1, 2, 4.3)
    nsteps = (0, 1, 2, 4)
    bases = (2.0, 1.1) if dtype in (torch.int8, torch.uint8) else (None, 2.0, 3.0, 1.1, 5.0)
    is_start_end_tensors = ((True, True), (True, False), (False, True))
    make_arg = partial(torch.tensor, device=device)
    for (start, end, nstep, base, (is_start_tensor, is_end_tensor)) in product(starts, ends, nsteps, bases, is_start_end_tensors):
        if dtype == torch.uint8 and end < 0 or start < 0:
            continue
        if nstep == 1 and isinstance(start, float) and (not (dtype.is_complex or dtype.is_floating_point)):
            continue
        tensor_options = {'dtype': dtype, 'device': device}
        if is_start_tensor:
            start = make_arg(start, dtype=torch.float32 if isinstance(start, float) else torch.int64)
        if is_end_tensor:
            end = make_arg(end, dtype=torch.float32 if isinstance(end, float) else torch.int64)
        if base is None:
            yield SampleInput(start, args=(end, nstep), kwargs=tensor_options)
        else:
            yield SampleInput(start, args=(end, nstep, base), kwargs=tensor_options)
    yield SampleInput(1, args=(3, 1, 2.0))

def sample_inputs_isclose(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs)
    rtols = [0.0, 1e-07]
    atols = [0.0, 1e-07]
    equal_nans = [False, True]
    products = product(rtols, atols, equal_nans)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for (rtol, atol, equal_nan) in products:
        lhs = make_arg((S, S), **op.lhs_make_tensor_kwargs)
        rhs = make_arg((S, S), **op.rhs_make_tensor_kwargs)
        yield SampleInput(lhs, args=(rhs,), kwargs=dict(rtol=rtol, atol=atol, equal_nan=equal_nan))

def error_inputs_isclose(op, device, **kwargs):
    if False:
        while True:
            i = 10
    make_float_arg = partial(make_tensor, device=device, dtype=torch.float, requires_grad=False)
    yield ErrorInput(SampleInput(make_float_arg(()), args=(make_float_arg(()),), kwargs={'rtol': -0.4}), error_type=RuntimeError, error_regex='rtol must be greater than or equal to zero')
    yield ErrorInput(SampleInput(make_float_arg(()), args=(make_float_arg(()),), kwargs={'atol': -0.4}), error_type=RuntimeError, error_regex='atol must be greater than or equal to zero')

def sample_inputs_t(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((1, 2)))
    yield SampleInput(make_arg((2,)))
    yield SampleInput(make_arg(()))

def sample_inputs_mm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_arg_conj(size):
        if False:
            for i in range(10):
                print('nop')
        return make_arg(size).conj().requires_grad_(requires_grad)
    (first_shape, second_shape) = ((S, M), (M, S))
    yield SampleInput(make_arg(first_shape), args=(make_arg(second_shape),))
    if dtype.is_complex:
        yield SampleInput(make_arg(first_shape), args=(make_arg_conj(second_shape),))

def sample_inputs_addmm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    alpha_val = kwargs.get('alpha', 2 + 3j if dtype.is_complex else 0.6)
    beta_val = kwargs.get('beta', 1 + 2j if dtype.is_complex else 0.2)
    tests_list = [((2, 3), (2, 2), (2, 3), False)]
    tests_with_lhs_broadcasting = [((1,), (2, 2), (2, 3), True), ((), (2, 2), (2, 3), True)]
    test_cases = tests_list + tests_with_lhs_broadcasting
    kwargs = dict(alpha=alpha_val, beta=beta_val)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for (shape_a, shape_b, shape_c, broadcasts_input) in test_cases:
        yield SampleInput(make_arg(shape_a), make_arg(shape_b), make_arg(shape_c), **kwargs).with_metadata(broadcasts_input=broadcasts_input)
    if dtype.is_complex:
        shape = (3, 3)
        yield SampleInput(make_arg(shape), make_arg(shape, requires_grad=False).mH.requires_grad_(requires_grad), make_arg(shape), **kwargs)
        yield SampleInput(make_arg(shape), make_arg(shape), make_arg(shape, requires_grad=False).mH.requires_grad_(requires_grad), **kwargs)

def sample_inputs_sparse_sampled_addmm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    alpha = 2 + 3j if dtype.is_complex else 0.6
    beta = 1 + 2j if dtype.is_complex else 0.2
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for (m, n, k) in itertools.product([0, 5], repeat=3):
        yield SampleInput(torch.eye(m, n, device=device, dtype=dtype).to_sparse_csr().requires_grad_(requires_grad), make_arg((m, k)), make_arg((k, n)), alpha=alpha, beta=beta)

def sample_inputs_sparse_mm_reduce(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    reductions = ['sum', 'mean', 'amax', 'amin']
    for (m, k, reduce) in product([5, 7], [3, 11], reductions):
        yield SampleInput(torch.eye(m, m).to(device=device, dtype=dtype).to_sparse_csr().requires_grad_(requires_grad), make_arg((m, k)), reduce)

def sample_inputs_mv(self, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(S, M), make_arg(M))

def sample_inputs_bmm(self, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg(M, S, M), make_arg(M, M, S))

def sample_inputs_dot_vdot(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_arg_conj(size):
        if False:
            i = 10
            return i + 15
        return make_arg(size).conj().requires_grad_(requires_grad)
    yield SampleInput(make_arg((S,)), make_arg((S,)))
    if dtype.is_complex:
        yield SampleInput(make_arg((S,)), make_arg_conj((S,)))

def error_inputs_dot_vdot(op_info, device, is_ref=False, **kwargs):
    if False:
        print('Hello World!')
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    if not is_ref:
        yield ErrorInput(SampleInput(make_input(1), args=(make_input(3, dtype=torch.float16),)), error_regex='dot : expected both vectors to have same dtype')
    yield ErrorInput(SampleInput(make_input(1, 1), args=(make_input(3),)), error_regex='1D tensors expected')
    yield ErrorInput(SampleInput(make_input(9), args=(make_input(3),)), error_regex='inconsistent tensor size')
    if device != 'cpu' and (not is_ref):
        yield ErrorInput(SampleInput(make_input(3), args=(make_input(3, device='cpu'),)), error_regex='Expected all tensors to be on the same device')

def sample_inputs_addmv(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases = (((S,), (S, M), (M,), 1, 1, False), ((S,), (S, M), (M,), 0.2, 0.6, False))
    test_cases_with_broadcast = (((1,), (S, M), (M,), 1, 1, True), ((1,), (S, M), (M,), 0.2, 0.6, True), ((), (S, M), (M,), 1, 1, True), ((), (S, M), (M,), 0.2, 0.6, True))
    cases = test_cases + test_cases_with_broadcast
    for (size, mat, vec, beta, alpha, broadcasts_input) in cases:
        yield SampleInput(make_arg(size), args=(make_arg(mat), make_arg(vec)), kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=broadcasts_input)

def sample_inputs_addbmm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = [((S, M), (S, S, S), (S, S, M), 1, 1, False), ((1,), (S, S, S), (S, S, M), 1, 1, True), ((S, M), (S, S, S), (S, S, M), 0.6, 0.2, False), ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True), ((), (S, S, S), (S, S, M), 1, 1, True), ((), (S, S, S), (S, S, M), 0.6, 0.2, True)]
    for (input_shape, batch1_shape, batch2_shape, beta, alpha, is_broadcasting) in test_cases:
        if dtype.is_complex:
            (beta_complex, alpha_complex) = (beta * (1 + 2j), alpha * (2 + 3j))
            yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)), kwargs=dict(beta=beta_complex, alpha=alpha_complex), broadcasts_input=is_broadcasting)
        yield SampleInput(make_arg(input_shape), args=(make_arg(batch1_shape), make_arg(batch2_shape)), kwargs=dict(beta=beta, alpha=alpha), broadcasts_input=is_broadcasting)

def sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    test_cases = [(((S, S), (S, S), (S, S)), False), (((S, S), (S, 1), (1, S)), False), (((1,), (S, S, 1), (1, S)), True), (((), (), ()), False), (((S, S), (), ()), True), (((), (S, S, 1), (1, S)), True)]
    for (input_args, broadcasts_input) in test_cases:
        args = tuple((make_arg(arg, exclude_zero=True) if isinstance(arg, tuple) else arg for arg in input_args))
        yield SampleInput(*args).with_metadata(broadcasts_input=broadcasts_input)
        args = tuple((make_arg(arg, exclude_zero=True) if isinstance(arg, tuple) else arg for arg in input_args))
        yield SampleInput(*args, value=3.14 if dtype.is_floating_point or dtype.is_complex else 3).with_metadata(broadcasts_input=broadcasts_input)

def reference_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    yield from sample_inputs_addcmul_addcdiv(op_info, device, dtype, requires_grad, **kwargs)
    supported_dtypes = op_info.supported_dtypes(device)
    make_arg = partial(make_tensor, device=device, requires_grad=requires_grad)
    types = ((torch.float64, torch.complex128), (torch.bfloat16, torch.float32))
    values = (None, True, False, 3.14, 3, 1.0, 1, 0.0, 0, -3.14, -3, 3.14 + 2.71j)
    for ((type2, type3), value) in product(types, values):
        if type2 not in supported_dtypes or type3 not in supported_dtypes:
            continue
        if type(value) is complex and type2 is not torch.complex128:
            continue
        arg1 = make_arg([5, 5], dtype=dtype)
        arg2 = make_arg([5, 5], dtype=type2)
        arg3 = make_arg([1, 5], dtype=type3)
        if value is not None:
            yield SampleInput(arg1, args=(arg2, arg3), kwargs=dict(value=value))
        else:
            yield SampleInput(arg1, args=(arg2, arg3))

def sample_inputs_baddbmm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    test_cases = [((S, S, M), (S, S, S), (S, S, M), 1, 1, False), ((1,), (S, S, S), (S, S, M), 1, 1, True), ((S, S, M), (S, S, S), (S, S, M), 0.6, 0.2, False), ((1,), (S, S, S), (S, S, M), 0.6, 0.2, True), ((), (S, S, S), (S, S, M), 1, 1, True), ((), (S, S, S), (S, S, M), 0.6, 0.2, True)]
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    for (input_shape, batch1_shape, batch2_shape, alpha, beta, broadcasts_input) in test_cases:
        yield SampleInput(make_arg(input_shape), make_arg(batch1_shape), make_arg(batch2_shape), beta=beta, alpha=alpha).with_metadata(broadcasts_input=broadcasts_input)
        if dtype.is_complex:
            yield SampleInput(make_arg(input_shape), make_arg(batch1_shape), make_arg(batch2_shape), beta=beta * (1 + 2j), alpha=alpha * (2 + 3j)).with_metadata(broadcasts_input=broadcasts_input)
    if dtype.is_complex:
        shapes = [(S, S, S), (S, M, S), (S, S, M)]
        args = tuple((make_arg(s) for s in shapes))
        yield SampleInput(args[0].transpose_(-1, 1), args[1].transpose(-1, 1).conj().requires_grad_(requires_grad), args[2].transpose(-1, 1).conj().requires_grad_(requires_grad), beta=beta * (1 + 2j), alpha=alpha * (2 + 3j))

def sample_inputs_multilabel_soft_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes = ((S,), (S, S))
    for shape in shapes:
        yield SampleInput(_make_tensor(shape), args=(_make_tensor(shape, requires_grad=False),), kwargs={})
        yield SampleInput(_make_tensor(shape), args=(_make_tensor(shape, requires_grad=False),), kwargs={'weight': _make_tensor(shape, requires_grad=False)})

def sample_inputs_addr(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    yield SampleInput(make_arg(S, M), make_arg(S), make_arg(M))
    yield SampleInput(make_arg(), make_arg(S), make_arg(M)).with_metadata(broadcasts_input=True)
    if dtype.is_complex:
        (alpha, beta) = (0.1 + 0.3j, 0.4 + 0.6j)
    elif dtype.is_floating_point:
        (alpha, beta) = (0.2, 0.6)
    else:
        (alpha, beta) = (2, 3)
    yield SampleInput(make_arg(S, M), make_arg(S), make_arg(M), beta=beta, alpha=alpha)
    yield SampleInput(make_arg(), make_arg(S), make_arg(M), beta=beta, alpha=alpha).with_metadata(broadcasts_input=True)
    if dtype.is_floating_point and (not requires_grad):
        tensor_options = dict(device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(torch.tensor([[math.nan]], **tensor_options), torch.tensor([0.0], **tensor_options), torch.tensor([0.0], **tensor_options), beta=0.0, alpha=0.0).with_metadata(broadcasts_input=True)
        yield SampleInput(torch.tensor([[0.0]], **tensor_options), torch.tensor([math.nan], **tensor_options), torch.tensor([math.nan], **tensor_options), beta=0.0, alpha=0.0).with_metadata(broadcasts_input=True)

def sample_inputs_zero_(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = ((), (S, S, S), (S,))
    for shape in cases:
        yield SampleInput(make_arg(shape))

def sample_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_weight = partial(_make_tensor, requires_grad=False)
    inputs = (((), make_target([], low=0, high=1), {}), ((S,), make_target([], low=0, high=S), {'p': 1}), ((S,), make_target([1], low=0, high=S), {'p': 2}), ((S, M), make_target([S], low=0, high=M), {'margin': 1.0}), ((S, M), make_target([S], low=0, high=M), {'margin': -3.14}), ((M, S), make_target([M], low=0, high=S), {'weight': None}), ((M, S), make_target([M], low=0, high=S), {'weight': make_weight([S], low=-10.0, high=10.0)}), ((M, S), make_target([M], low=0, high=S), {'reduction': 'none'}), ((M, S), make_target([M], low=0, high=S), {'reduction': 'mean'}), ((M, S), make_target([M], low=0, high=S), {'reduction': 'sum'}))
    for (input_shape, target, kwargs) in inputs:
        yield SampleInput(_make_tensor(input_shape), args=(target,), kwargs=kwargs)

def reference_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_multi_margin_loss(op_info, device, dtype, requires_grad, **kwargs)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_weight = partial(_make_tensor, requires_grad=False)
    inputs = (((), make_target([], low=0, high=1)), ((S,), make_target([], low=0, high=S)), ((S,), make_target([1], low=0, high=S)), ((M, S), make_target([M], low=0, high=S)))
    ps = (1, 2)
    margins = (0, 7, -3.14)
    weights = (False, True)
    reductions = (None, 'none', 'mean', 'sum')
    for ((input_shape, target), p, margin, weight, reduction) in product(inputs, ps, margins, weights, reductions):
        input = _make_tensor(input_shape)
        weight_shape = [input.size(-1)] if input.ndim > 0 else [1]
        weight = make_weight(weight_shape, low=-10.0, high=10.0) if weight else None
        kwargs = {'p': p, 'margin': margin, 'weight': weight}
        if reduction is not None:
            kwargs['reduction'] = reduction
        yield SampleInput(input, args=(target,), kwargs=kwargs)

def error_inputs_multi_margin_loss(op, device, **kwargs):
    if False:
        return 10
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='abc is not a valid value for reduction')
    yield ErrorInput(SampleInput(make_input(5, 0), args=(make_input(5),), kwargs={}), error_type=RuntimeError, error_regex='Expected non-empty vector or matrix with optional 0-dim batch size, but got: \\[5, 0\\]')
    yield ErrorInput(SampleInput(make_input(0), args=(make_input(5),), kwargs={}), error_type=RuntimeError, error_regex='Expected non-empty vector or matrix with optional 0-dim batch size, but got: \\[0\\]')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4),), kwargs={}), error_type=RuntimeError, error_regex='inconsistent target size, expected 5 but got \\[5, 4\\]')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5),), kwargs={}), error_type=RuntimeError, error_regex='expected scalar type Long but found Float')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5),), kwargs={'weight': make_input(())}), error_type=ValueError, error_regex='weight must be one-dimensional')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5),), kwargs={'weight': make_input(5, 4)}), error_type=ValueError, error_regex='weight must be one-dimensional')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5),), kwargs={'weight': make_input(5)}), error_type=RuntimeError, error_regex='inconsistent weight size, expected 4 but got \\[5\\]')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5),), kwargs={'p': 3}), error_type=ValueError, error_regex='only p == 1 and p == 2 supported')

def sample_inputs_logsumexp(self, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    inputs = (((), (0,), True), ((S, S), (1,), True), ((S, S), (1,), False), ((S, S), (-2,), False), ((S, S), (0, 1), False))
    lows = (None, 1000.0, 1000000.0) if dtype in (torch.float32, torch.float64) else (None,)
    for low in lows:
        high = low * 2 if low is not None else None
        for (shape, dim, keepdim) in inputs:
            t = make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)
            yield SampleInput(t, dim, keepdim)

def reference_inputs_logsumexp(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_logsumexp(op, device, dtype, requires_grad, **kwargs)
    t = torch.tensor([20, 30, 100], dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t, 0, False)
    t = torch.tensor((), dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t, 0, False)
    t = torch.tensor(float('inf'))
    yield SampleInput(t, 0, True)

def sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    inputs = [((), {}), ((S, S), {}), ((0, S, 0), {}), ((S,), {'dtype': dtype, 'device': device}), ((S,), {'dtype': torch.double}), ((S,), {'device': 'cpu'}), ((S,), {'dtype': torch.double, 'device': 'cpu'})]
    if torch.cuda.is_available():
        inputs.append(((S,), {'device': 'cuda'}))
    for (shape, kwargs) in inputs:
        t = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(t, **kwargs)

def reference_inputs_like_fns(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_like_fns(op, device, dtype, requires_grad, **kwargs)
    cases = ((), (0,), (1, 0), (1, 1, 4, 5), (5, 3, 0, 1), (1, 4, 3, 1, 1))
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape in cases:
        yield SampleInput(make_arg(shape))
        yield SampleInput(make_arg(shape).transpose(0, -1))
        yield SampleInput(make_arg(shape, noncontiguous=True))
        yield SampleInput(make_arg(shape, noncontiguous=True).transpose(0, -1))

def sample_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    inputs = (([], make_target([], low=0, high=1), {}), ([S], make_target([S], low=0, high=S), {}), ([M, S], make_target([M, S], low=0, high=S), {}), ([M, S], make_target([M, S], low=0, high=S), {'reduction': 'none'}), ([M, S], make_target([M, S], low=0, high=S), {'reduction': 'mean'}), ([M, S], make_target([M, S], low=0, high=S), {'reduction': 'sum'}))
    for (shape, target, kwargs) in inputs:
        yield SampleInput(_make_tensor(shape), args=(target,), kwargs=kwargs)

def reference_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    yield from sample_inputs_multilabel_margin_loss(op_info, device, dtype, requires_grad, **kwargs)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_target = partial(_make_tensor, dtype=torch.long, requires_grad=False)
    make_target_tensor = partial(torch.tensor, device=device, dtype=torch.long, requires_grad=False)
    inputs = (([], make_target([], low=-1, high=1)), ([S], make_target([S], low=-1, high=S)), ([M, S], make_target([M, S], low=-1, high=S)), ([], make_target_tensor(-1)), ([7], make_target_tensor([2, 0, 6, -1, 4, -1, 6])), ([4, 5], make_target_tensor([[4, -1, 0, -1, 2], [0, 0, 4, 1, 4], [-1, 3, -1, 1, 0], [4, 3, 2, 1, 0]])))
    reductions = (None, 'none', 'mean', 'sum')
    for ((shape, target), reduction) in product(inputs, reductions):
        kwargs = {}
        if reduction is not None:
            kwargs['reduction'] = reduction
        yield SampleInput(_make_tensor(shape), args=(target,), kwargs=kwargs)

def error_inputs_multilabel_margin_loss(op, device, **kwargs):
    if False:
        print('Hello World!')
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='abc is not a valid value for reduction')
    yield ErrorInput(SampleInput(make_input(5, 0), args=(make_input(5, 4),), kwargs={}), error_type=RuntimeError, error_regex='Expected non-empty vector or matrix with optional 0-dim batch size, but got: \\[5, 0\\]')
    yield ErrorInput(SampleInput(make_input(0), args=(make_input(0),), kwargs={}), error_type=RuntimeError, error_regex='Expected non-empty vector or matrix with optional 0-dim batch size, but got: \\[0\\]')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(4),), kwargs={}), error_type=RuntimeError, error_regex='inconsistent target size: \\[4\\] for input of size: \\[5, 4\\]')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(()),), kwargs={}), error_type=RuntimeError, error_regex='inconsistent target size: \\[\\] for input of size: \\[5, 4\\]')

def get_independent_tensor(tensor):
    if False:
        return 10
    return tensor.clone().requires_grad_(tensor.requires_grad)

def sample_inputs_randint(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    low = 2
    high = 10
    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        sample.kwargs.setdefault('device', device)
        yield SampleInput(high, sample.input.shape, *sample.args, **sample.kwargs)
        yield SampleInput(low, high, sample.input.shape, *sample.args, **sample.kwargs)

def sample_inputs_randint_like(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    low = 2
    high = 10
    for sample in sample_inputs_like_fns(self, device, dtype, requires_grad, **kwargs):
        yield SampleInput(sample.input, high, *sample.args, **sample.kwargs)
        yield SampleInput(get_independent_tensor(sample.input), low, high, *sample.args, **sample.kwargs)

def sample_inputs_margin_ranking_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes = ((), (S,), (S, S), (S, S, S))
    margins = (0.0, 1.0)
    reductions = ('sum', 'mean', 'none')
    for shape in shapes:
        for (margin, reduction) in product(margins, reductions):
            kwargs = {'margin': margin, 'reduction': reduction}
            yield SampleInput(_make_tensor(shape), args=(_make_tensor(shape, requires_grad=False), _make_tensor(shape, requires_grad=False)), kwargs=kwargs)

def reference_inputs_margin_ranking_loss(op, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    yield from sample_inputs_margin_ranking_loss(op, device, dtype, requires_grad, **kwargs)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for reduction in ('sum', 'mean', 'none'):
        if dtype.is_floating_point:
            inp1 = make_input((10,))
            inp1[2] = float('nan')
            inp2 = make_input((10,))
            inp2[4] = float('nan')
            target = make_input((10,))
            inp2[9] = float('nan')
            yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})
            inp1 = make_input((10,))
            inp2[1] = float('inf')
            inp2 = make_input((10,))
            inp2[4] = float('inf')
            target = make_input((10,))
            inp2[7] = float('inf')
            yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})
        inp1 = make_input((5, 2))
        inp2 = make_input((5, 1))
        target = make_input((1, 2))
        yield SampleInput(inp1, args=(inp2, target), kwargs={'reduction': reduction})

def error_inputs_margin_ranking_loss(op, device, **kwargs):
    if False:
        i = 10
        return i + 15
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4), make_input(5, 4)), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='is not a valid value')
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4), make_input(5))), error_regex='margin_ranking_loss : All input tensors should')

def sample_inputs_new_fns(self, device, dtype, requires_grad, *, is_strided=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    inputs = [((), (), (), {}), ((S, S), (2, 0), (3, 4), {}), ((0, S, 0), (3, 2, 2), (1, 2, 3), {}), ((S,), (2, 3), (7, 8), {'dtype': dtype, 'device': device}), ((S,), (10,), (S,), {'dtype': torch.double}), ((S,), (1, 1, 12), (S, L, M), {'device': 'cpu'}), ((S,), (2, 2, 2), (L, M, S), {'dtype': torch.double, 'device': 'cpu'})]
    if torch.cuda.is_available():
        inputs.append(((S,), (7, 2), (3, 4), {'device': 'cuda'}))
    for (input_shape, output_shape, strides, kwargs) in inputs:
        t = make_tensor(input_shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        if is_strided:
            yield SampleInput(t, output_shape, strides, **kwargs)
        else:
            yield SampleInput(t, output_shape, **kwargs)

def sample_inputs_empty_strided(op, device, dtype, requires_grad=False, **kwargs):
    if False:
        i = 10
        return i + 15
    inputs = [((), (), {'dtype': dtype, 'device': device}), ((S,), (4,), {'dtype': dtype, 'device': device}), ((S, S), (2, 1), {'dtype': dtype, 'device': device}), ((S, S, S), (2, 0, 1), {'dtype': dtype, 'device': device})]
    for (shape, strides, kwargs) in inputs:
        yield SampleInput(shape, strides, requires_grad=requires_grad, **kwargs)

def sample_inputs_empty(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    cases = ((), (0,), (1,), (1, 3, 5), (5, 3, 1), (1, 0, 5, 1))
    for case in cases:
        yield SampleInput(case, device=device, dtype=dtype, requires_grad=requires_grad)

def sample_inputs_empty_permuted(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    cases = ((), (0,), (1,), (1, 3, 5), (5, 3, 1), (1, 0, 5, 1))
    for case in cases:
        for layout in itertools.permutations(range(len(case))):
            yield SampleInput(case, layout, device=device, dtype=dtype, requires_grad=requires_grad)

def error_inputs_empty_permuted(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    yield ErrorInput(SampleInput((2,), args=((0, 1),)), error_type=RuntimeError, error_regex='Number of dimensions in size does not match the length of the physical_layout')
    yield ErrorInput(SampleInput((2,), args=((3,),)), error_type=RuntimeError, error_regex='Dimension out of range')
    yield ErrorInput(SampleInput((2, 3), args=((0, 0),)), error_type=RuntimeError, error_regex='Duplicate dim not allowed')

def sample_inputs_scalar_tensor(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    vals = (-5, 0, 1)
    for item in vals:
        yield SampleInput(item, device=device, dtype=dtype, requires_grad=requires_grad)

def sample_inputs_eye(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    sizes = (None, 0, 1, 2, 3, 4, 7, L, M, S)
    for (n, m) in product(sizes, sizes):
        if n is None:
            continue
        _kwargs = {'device': device, 'dtype': dtype, 'requires_grad': requires_grad}
        if m is None:
            yield SampleInput(n, args=(), kwargs=_kwargs)
        else:
            yield SampleInput(n, args=(m,), kwargs=_kwargs)

def error_inputs_eye(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    _kwargs = {'device': device, 'dtype': torch.float32}
    yield ErrorInput(SampleInput(-1, args=(), kwargs=_kwargs), error_regex='n must be greater or equal to 0, got -1')
    yield ErrorInput(SampleInput(-7, args=(42,), kwargs=_kwargs), error_regex='n must be greater or equal to 0, got -7')
    yield ErrorInput(SampleInput(0, args=(-3,), kwargs=_kwargs), error_regex='m must be greater or equal to 0, got -3')

def sample_inputs_new_full(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')

    def get_val(dtype):
        if False:
            return 10
        return make_tensor([], dtype=dtype, device='cpu').item()
    for sample in sample_inputs_new_fns(self, device, dtype, requires_grad, **kwargs):
        use_dtype = sample.kwargs['dtype'] if 'dtype' in sample.kwargs else dtype
        yield SampleInput(sample.input, *sample.args, get_val(use_dtype), **sample.kwargs)

def sample_inputs_full_like(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10

    def get_val(dtype):
        if False:
            print('Hello World!')
        return make_tensor([], dtype=dtype, device='cpu').item()
    inputs = [((), get_val(dtype), {}), ((S, S), get_val(dtype), {}), ((0, S, 0), get_val(dtype), {}), ((S,), get_val(dtype), {'dtype': dtype, 'device': device}), ((S,), get_val(torch.double), {'dtype': torch.double}), ((S,), get_val(dtype), {'device': 'cpu'}), ((S,), get_val(torch.double), {'dtype': torch.double, 'device': 'cpu'})]
    if torch.cuda.is_available():
        inputs.append(((S,), get_val(dtype), {'device': 'cuda'}))
    for (shape, fill_value, kwargs) in inputs:
        t = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(t, fill_value, **kwargs)

def sample_inputs_multinomial(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    cases = [([3], 3, {}), ([10], 3, {}), ([3, 10], 3, {}), ([3], 3, dict(replacement=False)), ([3], 3, dict(replacement=True)), ([3, 4], 4, dict(replacement=True)), ([3, 4], 4, dict(replacement=False))]
    for (shape, num_samples, kwargs) in cases:
        t = make_tensor(shape, dtype=dtype, device=device, low=0, high=None, requires_grad=requires_grad)
        yield SampleInput(t, num_samples, **kwargs)

def sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs):
    if False:
        while True:
            i = 10

    def get_value_or_make_tensor(value_or_shape):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(value_or_shape, list):
            return make_tensor(value_or_shape, dtype=dtype, device=device, low=0, high=None, requires_grad=requires_grad)
        return value_or_shape
    for (value_or_mean_shape, value_or_std_shape, kwargs) in cases:
        mean = get_value_or_make_tensor(value_or_mean_shape)
        std = get_value_or_make_tensor(value_or_std_shape)
        yield SampleInput(mean, std, **kwargs)

def sample_inputs_normal_tensor_first(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    cases = [([], [], {}), ([3], [3], {}), ([3, 4, 2], [3, 4, 2], {}), ([2, 3], 1.1, {}), ([1, 2, 3], [5, 2, 3], {})]
    return sample_inputs_normal_common(self, device, dtype, requires_grad, cases, **kwargs)

def sample_inputs_normal_tensor_second(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield SampleInput(1.6, 0.3, [2, 3], dtype=dtype, device=device)
    yield SampleInput(1.6, 0.3, [2, 2, 2], dtype=dtype, layout=torch.strided, device=device)
    yield SampleInput(2.7, make_tensor([4, 3], dtype=dtype, device=device, low=0, high=None, requires_grad=requires_grad))

def sample_inputs_bernoulli(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    shapes = [[3], [], [0, 3], [2, 3, 4]]
    for shape in shapes:
        t = make_tensor(shape, dtype=dtype, device=device, low=0, high=1, requires_grad=requires_grad)
        yield SampleInput(t)

def error_inputs_bernoulli(op_info, device, **kwargs):
    if False:
        return 10
    x = torch.rand((1,), device=device).expand((6,))
    err_msg = 'unsupported operation'
    yield ErrorInput(SampleInput(torch.rand_like(x), kwargs={'out': x}), error_regex=err_msg)

def sample_inputs_logcumsumexp(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    inputs = (((S, S, S), 0), ((S, S, S), 1), ((), 0))
    for large_number in (True, False):
        for (shape, dim) in inputs:
            t = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
            if large_number and t.dim() > 0:
                t[0] = 10000
            yield SampleInput(t, dim)

def sample_inputs_trace(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield SampleInput(make_tensor((S, S), dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad))

def error_inputs_trace(op, device):
    if False:
        for i in range(10):
            print('nop')
    yield ErrorInput(SampleInput(make_tensor((3, 4, 5), dtype=torch.float32, device=device)), error_regex='expected a matrix')

def sample_inputs_renorm(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (2, 1, 0.5)), ((S, S, S), (2, -1, 0.5)), ((S, S, S), (1, 2, 3)), ((S, S, S), (float('inf'), 2, 0.5)))
    for (shape, args) in cases:
        yield SampleInput(make_arg(shape), args=args)

def sample_inputs_transpose_swapdims(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((1, 2, 3), (-1, -2)), ((1, 2, 3), (-1, 2)), ((1, 2, 3), (1, -2)), ((1, 2, 3), (1, 2)), ((), (0, 0)), ((1,), (0, 0)), ((M, M), (0, 1)), ((S, S, S), (2, 0)))
    for (shape, args) in cases:
        yield SampleInput(make_arg(shape), args=args)

def _numpy_ref_transpose(a, dim0, dim1):
    if False:
        print('Hello World!')
    if a.ndim <= 1:
        return a
    return np.swapaxes(a, dim0, dim1)

def sample_inputs_adjoint(self, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes = ((1, 2, 3), (M, M), (S, S, S), (S, M, S), (M, S, M, S))
    return (SampleInput(make_arg(shape)) for shape in shapes)

def sample_inputs_T(self, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes = ((M, M), (M, L))
    return (SampleInput(make_arg(shape)) for shape in shapes)

def error_inputs_T(self, device, has_ndims_error=False):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    if has_ndims_error:
        yield ErrorInput(SampleInput(make_arg(M)), error_regex='The use of `x\\.T` on tensors of dimension other than 0 or 2 to reverse their shape is not supported\\.')
        yield ErrorInput(SampleInput(make_arg(M, S, L)), error_regex='The use of `x\\.T` on tensors of dimension other than 0 or 2 to reverse their shape is not supported\\.')

def sample_inputs_singular_matrix_factors(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        return 10
    '\n    This function produces two tensors of shape (*, m, k) and (*, n, k) with k <= min(m, n).\n    Their matrix product could be used to generate tensor of shape (*, m, n) of rank k.\n    '
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    batches = [(), (0,), (2,), (1, 1)]
    size = [1, 5, 10]
    for (batch, m, n) in product(batches, size, size):
        for k in range(min(3, m, n)):
            a = make_arg((*batch, m, k))
            b = make_arg((*batch, n, k))
            yield SampleInput(a, b, **kwargs)

def sample_inputs_svd_lowrank(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    for sample in sample_inputs_singular_matrix_factors(op_info, device, dtype, requires_grad, **kwargs):
        (*batch, m, k) = sample.input.shape
        (*_, n, _) = sample.args[0].shape
        op_kwargs = {'q': k, 'M': None}
        yield clone_sample(sample, **op_kwargs)
        op_kwargs['M'] = make_tensor((*batch, m, n), dtype=dtype, device=device, requires_grad=requires_grad)
        yield clone_sample(sample, **op_kwargs)

def chunk_iter(iterable, size):
    if False:
        print('Hello World!')
    it = iter(iterable)
    while True:
        chunk = tuple(islice(it, size))
        if not chunk:
            break
        yield chunk

def sample_inputs_pca_lowrank(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        return 10
    samples = sample_inputs_svd_lowrank(op_info, device, dtype, requires_grad, **kwargs)
    for (s1, s2) in chunk_iter(samples, 2):
        del s1.kwargs['M']
        del s2.kwargs['M']
        s1.kwargs['center'] = False
        s2.kwargs['center'] = True
        yield s1
        yield s2

def np_sinc_with_fp16_as_fp32(x):
    if False:
        return 10
    if x.dtype == np.float16:
        return np.sinc(x.astype(np.float32))
    else:
        return np.sinc(x)

def sample_inputs_broadcast_to(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    test_cases = (((S, 1, 1), (S, S, S)), ((S, 1, S), (S, S, S)), ((S, 1), (S, S, S)), ((1,), (S, S, S)), ((1, S), (1, 1, S)), ((), ()), ((), (1, 3, 2)))
    return (SampleInput(make_tensor(size, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad), shape) for (size, shape) in test_cases)

def sample_inputs_broadcast_tensors(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases: Tuple[tuple] = (((3,), (1, 2, 1), (1, 1), (5, 1, 1)),)
    for (shape, *other_shapes) in test_cases:
        yield SampleInput(make_arg(shape), args=tuple((make_arg(s) for s in other_shapes)))

def reference_inputs_broadcast_tensors(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_broadcast_tensors(op, device, dtype, requires_grad, **kwargs)
    m = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    n = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad, noncontiguous=True)
    cases = (((), (1, 1), (1, 1, 7, 1), (3, 1, 1)), ((3, 5, 6), (1, 3, 5, 6), (1, 1, 1, 1, 6), (8, 3, 5, 6)))
    for (a, b, c, d) in cases:
        yield SampleInput(m(a), args=(m(b), m(c), m(d)))
        yield SampleInput(n(a), args=(n(b), n(c), n(d)))

def sample_inputs_block_diag(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases: Tuple[tuple] = (((1, S), (2, S), (3, S)), ((S, 1), (S, 2), (S, 3)), ((1,), (2,), (3,)), ((2, S), (S,)))
    for (shape, *other_shapes) in test_cases:
        yield SampleInput(make_arg(shape), args=tuple((make_arg(s) for s in other_shapes)))
        if dtype == torch.complex32 or dtype == torch.complex64:
            non_complex_dtype = torch.float32 if dtype == torch.complex32 else torch.float64
            make_arg_non_complex = partial(make_tensor, dtype=non_complex_dtype, device=device, requires_grad=requires_grad)
            yield SampleInput(make_arg_non_complex(shape), args=tuple((make_arg(s) for s in other_shapes)))

def sample_inputs_cdist(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    small_S = 2
    test_cases = (((S, S, 2), (S, S + 1, 2)), ((S, S), (S, S)), ((S, S, S), (S, S, S)), ((3, 5), (3, 5)), ((2, 3, 5), (2, 3, 5)), ((1, 2, 3), (1, 2, 3)), ((1, 1), (S, 1)), ((0, 5), (4, 5)), ((4, 5), (0, 5)), ((0, 4, 5), (3, 5)), ((4, 5), (0, 3, 5)), ((0, 4, 5), (1, 3, 5)), ((1, 4, 5), (0, 3, 5)), ((small_S, small_S, small_S + 1, 2), (small_S, small_S, small_S + 2, 2)), ((small_S, 1, 1, small_S), (1, small_S, small_S)), ((1, 1, small_S), (small_S, 1, small_S, small_S)))
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for cm in ['use_mm_for_euclid_dist', 'donot_use_mm_for_euclid_dist']:
        for p in [0.0, 1.0, 2.0, 3.0, 0.5, 1.5, 2.5, float('inf')]:
            for (t1_size, t2_size) in test_cases:
                yield SampleInput(make_arg(t1_size), make_arg(t2_size), p, cm)

def _fill_np(a, value):
    if False:
        for i in range(10):
            print('nop')
    a = a.copy()
    a.fill(value)
    return a

def _fill_sample_kwargs(device, dtype, input):
    if False:
        for i in range(10):
            print('nop')
    if dtype is torch.bool:
        value = True
    else:
        value = 3
    return ({'value': value}, {'value': value})

def sample_inputs_comparison_ops(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_elementwise_binary(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    lhs = make_arg((S, S))
    yield SampleInput(lhs, args=(lhs.clone(),))

def sample_inputs_stack(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((3, 4), 1), ((1, 2, 1, 4), 3), ((0, 1, 0), 2))
    for (shape, num_tensors) in cases:
        tensors = []
        for _ in range(num_tensors):
            tensors.append(make_arg(shape))
        for dim in range(-1, len(shape) - 1):
            yield SampleInput(tensors, args=(dim,))

def sample_inputs_cat_concat(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[tuple, tuple, dict] = (((S, S), (S, S), {'dim': -1}), ((S, S), (S, S), {'dim': 1}), ((M, S), (S, S), {'dim': 0}), ((1, 2, 3), (1, 2, 3), {'dim': -2}), ((0,), (0,), {'dim': 0}), ((0,), (S, S), {'dim': 1}), ((0, S), (S, S), {'dim': 0}), ((1,), (1,), {}))
    for (input_shape1, input_shape2, kwargs) in cases:
        yield SampleInput([make_arg(input_shape1), make_arg(input_shape2)], kwargs=kwargs)
    yield SampleInput([make_arg((2, 2, 2, 2), memory_format=torch.channels_last)], args=(1,))

def error_inputs_cat(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput([make_arg((S, S)), make_arg((S, S))], kwargs={'out': make_arg((1, S)).expand((2 * S, S))}), error_regex='unsupported operation')
    yield ErrorInput(SampleInput([], kwargs={'dim': 1}), error_regex='non-empty list of Tensors')
    yield ErrorInput(SampleInput([make_arg((S, S, L, L)), make_arg((S, 0, L - 1, L))], kwargs={'dim': 1}), error_regex='Sizes of tensors must match except in dimension')
    yield ErrorInput(SampleInput([make_arg((S, 0, L - 1, L)), make_arg((S, S, L, L))], kwargs={'dim': 1}), error_regex='Sizes of tensors must match except in dimension')
    yield ErrorInput(SampleInput([make_arg((S - 1, 0)), make_arg((S, 0, L - 1, L))], kwargs={'dim': 1}), error_regex='Tensors must have same number of dimensions')
    yield ErrorInput(SampleInput([make_arg((S, 0, L - 1, L)), make_arg((S - 1, 0))], kwargs={'dim': 1}), error_regex='Tensors must have same number of dimensions')
    x = torch.zeros(0, device=device)
    y = torch.randn((4, 6), device=device)
    err_msg = 'the written-to tensor refer to a single memory location'
    yield ErrorInput(SampleInput((x, y), kwargs={'dim': 0, 'out': x}), error_regex=err_msg)
    yield ErrorInput(SampleInput((x, y), kwargs={'dim': 0, 'out': y}), error_regex=err_msg)
    z = torch.zeros((4, 6), device=device)
    yield ErrorInput(SampleInput((y, z), kwargs={'out': z[:2, :]}), error_regex=err_msg)
    if torch.device(device).type == 'cuda':
        x_cuda = make_tensor((3, 3), device=device, dtype=torch.float32)
        y_cpu = make_tensor((3, 3), device='cpu', dtype=torch.float32)
        yield ErrorInput(SampleInput((x_cuda, y_cpu)), error_regex='Expected all tensors to be on the same device')
    yield ErrorInput(SampleInput([make_arg((L, 1)), make_arg((L, 1, 1)), make_arg((L, 1, 1))]), error_regex='Tensors must have same number of dimensions')
    yield ErrorInput(SampleInput([make_arg((S, 1, M)), make_arg((S, 1, 1)), make_arg((S, M, 1))], kwargs={'dim': 1}), error_regex='Sizes of tensors must match')
    yield ErrorInput(SampleInput((make_arg((S, 1, 1)), None)), error_type=TypeError, error_regex='got None')
    yield ErrorInput(SampleInput([make_arg(()), make_arg(())]), error_regex='zero-dimensional.*cannot be concatenated')
    d = make_tensor((2, 3), device=device, dtype=torch.double)
    x = make_tensor((2, 3), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(x, kwargs={'out': d}), error_type=TypeError, error_regex='invalid combination of arguments')

def reference_inputs_cat(op, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    yield from sample_inputs_cat_concat(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    a = make_arg((3, 4, 2))
    b = make_arg((3, 2, 2), noncontiguous=True, dtype=torch.double)
    c = make_arg((3, 3, 2), dtype=torch.float16).permute(1, 0, 2)
    yield SampleInput((a, b, c), kwargs={'dim': 1})
    a = make_arg((0,))
    b = make_arg((3, 2, 2))
    yield SampleInput((a, b, a))
    yield SampleInput((a, a, a))

def _elementwise_type_promo_np(*args, type_promotion_kind):
    if False:
        i = 10
        return i + 15

    def _maybe_torch(x):
        if False:
            i = 10
            return i + 15
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        return x
    flattened = pytree.arg_tree_leaves(*args)
    transformed = tuple((_maybe_torch(a) for a in flattened))
    (result_dtype, _) = prims.utils.elementwise_dtypes(*transformed, type_promotion_kind=type_promotion_kind)
    return torch_to_numpy_dtype_dict[result_dtype]

def _cat_np(input_seq, dim=0):
    if False:
        print('Hello World!')
    inputs = tuple((a for a in input_seq if not (a.ndim == 1 and a.size == 0)))
    if len(inputs) == 0:
        np_dtype = _elementwise_type_promo_np(input_seq, type_promotion_kind=prims.utils.ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH)
        return np.empty(0, dtype=np_dtype)
    return np.concatenate(inputs, axis=dim)

def _floor_divide_np(a, b):
    if False:
        i = 10
        return i + 15
    dtype = _elementwise_type_promo_np(a, b, type_promotion_kind=prims.utils.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT)
    if isinstance(a, np.ndarray):
        a = a.astype(dtype)
    if isinstance(b, np.ndarray):
        b = b.astype(dtype)
    return np.floor_divide(a, b)

def sample_inputs_hstack_dstack_vstack(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    tensor_shapes = (((S,), (S,), (S,)), ((S, S), (S, S), (S, S)))
    for (s1, s2, s3) in tensor_shapes:
        tensors = (make_arg(s1), make_arg(s2), make_arg(s3))
        yield SampleInput(tensors)

def error_inputs_hstack_dstack_vstack(op, device):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=torch.int32, device=device, requires_grad=False)
    tensor_shapes = (((S,), (S, S, S, S), (S,)),)
    for (s1, s2, s3) in tensor_shapes:
        tensors = (make_arg(s1), make_arg(s2), make_arg(s3))
        yield ErrorInput(SampleInput(tensors), error_regex='Tensors must have same number of dimensions')
    yield ErrorInput(SampleInput(()), error_regex='expects a non-empty TensorList')

def sample_inputs_unbind(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    shape_dims = (((S,), 0), ((S, S), 0), ((S, S), 1), ((S, S), -1), ((S, 0, S), 0), ((S, S, S), 1))
    for (shape, dim) in shape_dims:
        yield SampleInput(make_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad), args=(dim,))

def error_inputs_unbind(op_info, device):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=torch.int32, device=device, requires_grad=False)
    yield ErrorInput(SampleInput(make_arg(()), args=(0,)), error_type=IndexError, error_regex='Dimension specified as 0 but tensor has no dimensions')
    yield ErrorInput(SampleInput(make_arg((2,)), args=(2,)), error_type=IndexError, error_regex='Dimension out of range')

def reference_unbind(t, dim):
    if False:
        return 10
    'A numpy implementation of torch.unbind'
    return tuple((s.squeeze(dim) for s in np.split(t, t.shape[dim], dim)))

def sample_inputs_gather(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    yield SampleInput(make_arg((M, S)), 0, gather_variable((S, S), 1, M, True, device=device))
    yield SampleInput(make_arg((M, S)), 1, gather_variable((M, S // 2), 0, S, True, device=device))
    yield SampleInput(make_arg((S,)), 0, torch.tensor([], dtype=torch.uint8, device=device))
    yield SampleInput(make_arg(()), 0, torch.tensor([0], dtype=torch.int64, device=device))
    yield SampleInput(make_arg(()), 0, torch.tensor(0, dtype=torch.int64, device=device))

def _fill_indices(idx, dim, dim_size, elems_per_row, m, n, o):
    if False:
        return 10
    for i in range(1 if dim == 0 else m):
        for j in range(1 if dim == 1 else n):
            for k in range(1 if dim == 2 else o):
                ii = [i, j, k]
                ii[dim] = slice(0, idx.size(dim) + 1)
                idx[tuple(ii)] = torch.randperm(dim_size)[0:elems_per_row]

def error_inputs_gather(op_info, device, **kwargs):
    if False:
        return 10
    src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
    bad_src = make_tensor((1, 1), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(bad_src, args=(1, idx)), error_regex='Size does not match at dimension 0')
    bad_idx = idx.to(torch.int32)
    yield ErrorInput(SampleInput(src, args=(1, bad_idx)), error_regex='Expected dtype int64 for index')
    src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
    out = torch.empty((2, 2), device=device, dtype=torch.float64)
    yield ErrorInput(SampleInput(src, args=(1, idx), kwargs={'out': out}), error_regex='Expected out tensor to have dtype')
    src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)
    idx = torch.tensor((0, 0), device=device, dtype=torch.long)
    yield ErrorInput(SampleInput(src, args=(1, idx)), error_regex='Index tensor must have the same number of dimensions')
    src = torch.tensor((1, 2), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 0), (1, 0)), device=device, dtype=torch.long)
    yield ErrorInput(SampleInput(src, args=(0, idx)), error_regex='Index tensor must have the same number of dimensions')
    if torch.device(device).type == 'cpu':
        src = torch.tensor(((1, 2), (3, 4)), device=device, dtype=torch.float32)
        idx = torch.tensor(((0, 23), (1, 0)), device=device, dtype=torch.long)
        yield ErrorInput(SampleInput(src, args=(1, idx)), error_regex='index 23 is out of bounds for dimension')
    x = torch.rand((1,), device=device).expand((3,))
    src = torch.rand((6,), device=device)
    ind = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)
    yield ErrorInput(SampleInput(src, args=(0, ind), kwargs=dict(out=x)), error_type=RuntimeError, error_regex='unsupported operation')
    yield ErrorInput(SampleInput(src, args=(0, ind), kwargs=dict(out=src)), error_type=RuntimeError, error_regex='unsupported operation')
    yield ErrorInput(SampleInput(ind.clone(), args=(0, ind[1:]), kwargs=dict(out=ind[:1])), error_type=RuntimeError, error_regex='unsupported operation')

def error_inputs_take(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    x = torch.rand((1,), device=device).expand((3,))
    src = torch.rand((6,), device=device)
    ind = torch.tensor([2, 1, 0], device=device, dtype=torch.int64)
    yield ErrorInput(SampleInput(src, args=(ind,), kwargs=dict(out=x)), error_type=RuntimeError, error_regex='unsupported operation')
    yield ErrorInput(SampleInput(src, args=(ind,), kwargs=dict(out=src)), error_type=RuntimeError, error_regex='unsupported operation')
    yield ErrorInput(SampleInput(ind.clone(), args=(ind[1:],), kwargs=dict(out=ind[:-1])), error_type=RuntimeError, error_regex='unsupported operation')

def error_inputs_scatter_and_scatter_add(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    src = make_tensor((2, 5), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 1), (1, 2)), device=device, dtype=torch.long)
    dst = torch.zeros((3, 5), device=device, dtype=torch.double)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_regex='Expected self.dtype to be equal to src.dtype')
    src = make_tensor((2, 5), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 1), (1, 2)), device=device, dtype=torch.int32)
    dst = torch.zeros((3, 5), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_regex='Expected dtype int64 for index')
    src = make_tensor((2, 5), device=device, dtype=torch.float32)
    idx = torch.tensor(((0, 1), (1, 2)), device=device, dtype=torch.long)
    dst = torch.zeros((3, 5, 3), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_regex='Index tensor must have the same number of dimensions as self tensor')
    src = make_tensor((2, 5, 2), device=device, dtype=torch.float32)
    idx = torch.tensor(((34, 1), (1, 2)), device=device, dtype=torch.long)
    dst = torch.zeros((3, 5), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_regex='Index tensor must have the same number of dimensions as src tensor')
    if torch.device(device).type == 'cpu':
        src = make_tensor((2, 5), device=device, dtype=torch.float32)
        idx = torch.tensor(((34, 1), (1, 2)), device=device, dtype=torch.long)
        dst = torch.zeros((3, 5), device=device, dtype=torch.float32)
        yield ErrorInput(SampleInput(dst, args=(0, idx, src)), error_regex='index 34 is out of bounds for dimension 0 with size 3')

def error_inputs_renorm(op_info, device, **kwargs):
    if False:
        print('Hello World!')
    zero_d = torch.randn((), device=device)
    yield ErrorInput(SampleInput(zero_d, args=(0.5, 0, 1.0)), error_type=RuntimeError, error_regex='needs at least 2 dimensions, got 0 dimensions')

def error_inputs_ormqr(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    zero_d = torch.randn((), device=device)
    yield ErrorInput(SampleInput(zero_d, args=(zero_d, zero_d)), error_type=RuntimeError, error_regex='input must have at least 2 dimensions')
    tensor_0 = torch.full((5, 0), 1, device=device)
    tensor_1 = torch.full((5,), 1, device=device)
    tensor_2 = torch.full((5, 5), 1, device=device)
    bool_3 = True
    bool_4 = True
    yield ErrorInput(SampleInput(tensor_0, args=(tensor_1, tensor_2, bool_3, bool_4)), error_type=RuntimeError, error_regex='tau.shape\\[-1\\] must be less than or equal to input.shape\\[-1\\]')

def error_inputs_diag(op_info, device, **kwargs):
    if False:
        print('Hello World!')
    zero_d = torch.randn((), device=device)
    yield ErrorInput(SampleInput(zero_d, args=(0,)), error_type=RuntimeError, error_regex='1D or 2D')
    zero_d = torch.randn(1, 1, 1, device=device)
    yield ErrorInput(SampleInput(zero_d, args=(0,)), error_type=RuntimeError, error_regex='1D or 2D')

def error_inputs_embedding(op_info, device, **kwargs):
    if False:
        return 10
    indices = torch.rand(2, 2, device=device).long()
    weights = [torch.tensor(1.0, device=device), torch.tensor(1.0, device=device).reshape(1, 1, 1)]
    for weight in weights:
        yield ErrorInput(SampleInput(weight, args=(indices,)), error_type=RuntimeError, error_regex="'weight' must be 2-D")

def error_inputs_t(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    yield ErrorInput(SampleInput(torch.randn(2, 3, 4, 5, device=device)), error_regex='expects a tensor with <= 2')

def error_inputs_multinomial(op_info, device, **kwargs):
    if False:
        return 10
    x = torch.empty(1, 2, 3, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(2,)), error_regex='prob_dist must be 1 or 2 dim')
    x = torch.empty(1, 2, dtype=torch.long, device=device)
    yield ErrorInput(SampleInput(x, args=(2,)), error_regex='multinomial only supports floating-point dtypes for input')
    x = torch.empty(1, 2, dtype=torch.double, device=device)
    y = torch.empty(1, 2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(2,), kwargs=dict(out=y)), error_regex='multinomial expects Long tensor out')
    x = torch.empty(2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(0,)), error_regex='cannot sample n_sample <= 0 samples')
    x = torch.empty(2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(-1,)), error_regex='cannot sample n_sample <= 0 samples')
    x = torch.empty(2, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(3, False)), error_regex='cannot sample n_sample > prob_dist')
    x = torch.empty(16777217, dtype=torch.double, device=device)
    yield ErrorInput(SampleInput(x, args=(3,)), error_regex='number of categories cannot exceed')
    inputs = ((1.0, -1.0, 1.0), (1.0, inf, 1.0), (1.0, -inf, 1.0), (1.0, 1.0, nan))
    err_msg1 = 'probability tensor contains either `inf`, `nan` or element < 0'
    err_msg2 = 'invalid multinomial distribution'
    rep_arg = (False, True) if torch.device(device).type == 'cpu' else (False,)
    for rep in rep_arg:
        kwargs = {'num_samples': 2, 'replacement': rep}
        for shape in inputs:
            yield ErrorInput(SampleInput(torch.tensor(shape), kwargs=kwargs), error_regex=err_msg1 if rep is False else err_msg2)
        x = torch.zeros(3, device=device)
        yield ErrorInput(SampleInput(x, kwargs=kwargs), error_regex=err_msg2)
        x = torch.zeros(3, 3, device=device)
        yield ErrorInput(SampleInput(x, kwargs=kwargs), error_regex=err_msg2)
        x[1, :] = 1
        yield ErrorInput(SampleInput(x, kwargs=kwargs), error_regex=err_msg2)

def error_inputs_gradient(op_info, device, **kwargs):
    if False:
        return 10
    for dtype in [torch.long, torch.float32, torch.complex64]:
        t = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9]], device=device, dtype=dtype)
        dim = (1, 0)
        spacing = [0.1]
        yield ErrorInput(SampleInput(t, kwargs=dict(spacing=spacing, dim=dim, edge_order=1)), error_type=RuntimeError, error_regex='torch.gradient expected spacing to be unspecified, a scalar ')
        yield ErrorInput(SampleInput(t, kwargs=dict(edge_order=3)), error_type=RuntimeError, error_regex='torch.gradient only supports edge_order=1 and edge_order=2.')
        dim = (1, 1)
        spacing = 0.1
        yield ErrorInput(SampleInput(t, kwargs=dict(spacing=spacing, dim=dim, edge_order=1)), error_type=RuntimeError, error_regex='dim 1 appears multiple times in the list of dims')
        dim = (0, 1)
        coordinates = [torch.tensor([1, 2, 4], device='cpu'), torch.tensor([1, 2, 4], device='meta')]
        yield ErrorInput(SampleInput(t, kwargs=dict(spacing=coordinates, dim=dim, edge_order=1)), error_type=RuntimeError, error_regex='torch.gradient expected each tensor to be on the same device,')
        yield ErrorInput(SampleInput(t, kwargs=dict(dim=3)), error_type=IndexError, error_regex='')
        t = torch.tensor([[1], [2], [3]])
        yield ErrorInput(SampleInput(t, kwargs=dict(edge_order=1)), error_type=RuntimeError, error_regex='torch.gradient expected each dimension size to be at least')
        t = torch.tensor([[1, 2], [3, 4]])
        yield ErrorInput(SampleInput(t, kwargs=dict(edge_order=2)), error_type=RuntimeError, error_regex='torch.gradient expected each dimension size to be at least')

def error_inputs_rrelu(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    input = make_tensor((S, S), device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(input, kwargs={'lower': 0.3, 'upper': 0.1}), error_regex='Lower bound should be less than or equal to the upper bound')

def error_inputs_masked_select(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    x = torch.rand((1,), device=device).expand((3,))
    y = torch.rand((6,), device=device)
    mask = torch.tensor([True, False, True, True, False, False], device=device)
    yield ErrorInput(SampleInput(y, args=(mask,), kwargs=dict(out=x)), error_type=RuntimeError, error_regex='unsupported operation')
    yield ErrorInput(SampleInput(y, args=(mask,), kwargs=dict(out=y)), error_type=RuntimeError, error_regex='unsupported operation')
    yield ErrorInput(SampleInput(mask.clone(), args=(mask,), kwargs=dict(out=mask)), error_type=RuntimeError, error_regex='unsupported operation')

def error_inputs_median(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    x = torch.tensor([[[[[[[[[[[[[[[[[[[[[[[[[nan], [nan]]]]]]]]]]]]]]]]]]]]]]]]], device=device)
    if device == 'cuda':
        yield ErrorInput(SampleInput(x, kwargs=dict(dim=-1)), error_type=RuntimeError, error_regex='CUDA Tensors cannot have more than 25 dimensions')
    else:
        return

def error_inputs_index_select(op_info, device, **kwargs):
    if False:
        print('Hello World!')
    x = torch.rand((1, 6), device=device).expand((2, 6))
    y = torch.rand((3, 6), device=device)
    ind = torch.tensor([0, 1], dtype=torch.int64, device=device)
    yield ErrorInput(SampleInput(y, args=(1, ind), kwargs=dict(out=x)), error_type=RuntimeError, error_regex='unsupported operation')

def error_inputs_index_add(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    result = torch.tensor([[1.0, 2.0], [4.0, 5.0], [7.0, 8.0]])
    source = torch.tensor([2.0, 4.0])
    yield ErrorInput(SampleInput(result, args=(0, torch.tensor([0, 2]), source)), error_type=RuntimeError, error_regex='source tensor shape must match self tensor shape, excluding the specified dimension. Got self.shape = \\[3, 2\\] source.shape = \\[2\\]')

def error_inputs_logcumsumexp(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    dim = 3
    srcs = [torch.randn(5, 2, device=device), torch.randn(0, 2, device=device)]
    for src in srcs:
        yield ErrorInput(SampleInput(src, args=(dim,)), error_type=IndexError, error_regex='Dimension out of range')

def sample_inputs_take_along_dim(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    yield SampleInput(make_arg((S, S)), gather_variable((S, S), 1, S, True, device=device), 0)
    yield SampleInput(make_arg((S, S)), gather_variable((1, S // 2), 0, S, True, device=device), 1)
    yield SampleInput(make_arg((1, S)), gather_variable((S, S // 2), 0, S, True, device=device), 1)
    yield SampleInput(make_arg((S, S)), gather_variable((S, S // 2), 0, S, True, device=device))

def error_inputs_aminmax_amax_amin(op_info, device, is_ref=False, **kwargs):
    if False:
        print('Hello World!')
    shape = (S, 0, S)
    err_msg_amax_amin = 'reduction'
    err_msg_aminmax = 'cannot compute aminmax over an empty dimension as the operation has no identity'
    if op_info.name in ['amax', 'amin', '_refs.amax', '_refs.amin']:
        yield ErrorInput(SampleInput(torch.rand(shape, device=device)), error_regex=err_msg_amax_amin)
    elif op_info.name in ['aminmax']:
        yield ErrorInput(SampleInput(torch.rand(shape, device=device)), error_regex=err_msg_aminmax)
    sizes = [1] * 65
    err_msg1 = 'only tensors with up to 64 dims are supported'
    yield ErrorInput(SampleInput(torch.randn(sizes, device=device), kwargs={'dim': -1}), error_regex=err_msg1)
    yield ErrorInput(SampleInput(torch.randn(sizes, device=device), kwargs={'dim': 64}), error_regex=err_msg1)
    if op_info.name in ['amax', 'amin', '_refs.amax', '_refs.amin']:
        dims = [(0, 0), (0, -4)]
        err_msg2 = 'in the list of dims'
        x = torch.randn(S, S, S, S, device=device)
        for dim in dims:
            yield ErrorInput(SampleInput(x, kwargs={'dim': dim}), error_regex=err_msg2)
    input5 = torch.randn(L, L, dtype=torch.float32, device=device)
    max_values = torch.empty(L, dtype=torch.float32, device=device)
    min_values = torch.empty(L, dtype=torch.double, device=device)
    illegal_values = torch.empty(L, dtype=torch.int, device=device)
    if is_ref:
        err_msg_amax_amin2 = "Attempting to cast from torch.float32 to out tensor with dtype torch.int32, but this can't be cast because it is not safe!"
    else:
        err_msg_amax_amin2 = "Expected the dtype for input and out to match, but got Float for input's dtype and Int for out's dtype."
    err_msg_aminmax2 = 'Expected out tensor to have dtype float, but got double instead'
    if op_info.name in ['amax', 'amin', '_refs.amax', '_refs.amin']:
        yield ErrorInput(SampleInput(input5, kwargs={'dim': 0, 'out': illegal_values}), error_regex=err_msg_amax_amin2)
    elif op_info.name in ['aminmax']:
        yield ErrorInput(SampleInput(input5, kwargs={'dim': 0, 'out': (max_values, min_values)}), error_regex=err_msg_aminmax2)
    err_msg3 = 'reduction'
    error_type = IndexError if 'refs' not in op_info.name else RuntimeError
    yield ErrorInput(SampleInput(torch.rand(shape, device=device), kwargs={'dim': 1}), error_type=error_type, error_regex=err_msg3)

def sample_inputs_aminmax(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    test_cases: Tuple[tuple, dict] = (((S, S, S), {}), ((S, S, S), {'dim': 1}), ((S, S, S), {'dim': 1, 'keepdim': True}), ((), {'dim': 0}), ((), {}), ((), {'dim': 0, 'keepdim': True}), ((S, 0, S), {'dim': 0}))
    for (shape, kwargs) in test_cases:
        yield SampleInput(make_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad), **kwargs)

def error_inputs_diff(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    t = torch.rand((1, 3), device=device)
    n = -1
    yield ErrorInput(SampleInput(t, args=(n,), kwargs=kwargs), error_type=RuntimeError, error_regex=f'order must be non-negative but got {n}')

def sample_inputs_diff(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_cases = (((1,), 0, None, None), ((S,), 0, None, None), ((S, 1), 0, None, None), ((S, 1), 1, None, None), ((S, S), 0, None, None), ((S, S), 1, None, None), ((S, S), 0, (1, S), (2, S)), ((S, S), 0, None, (2, S)), ((XS, XS, XS), 1, None, None), ((XS, XS, XS), 2, None, None), ((XS, XS, XS), 1, (XS, 1, XS), (XS, 1, XS)), ((XS, XS, XS), 2, (XS, XS, 1), (XS, XS, 1)), ((XS, XS, XS), 2, (XS, XS, XS), (XS, XS, XS)))
    sample_inputs = []
    for (size, dim, size_prepend, size_append) in test_cases:
        prepend_size = 0 if size_prepend is None else size_prepend[dim]
        append_size = 0 if size_append is None else size_append[dim]
        dim_size = size[dim] + prepend_size + append_size
        for n in range(dim_size):
            input_tensor = make_arg(size)
            prepend = make_arg(size_prepend) if size_prepend else None
            append = make_arg(size_append) if size_append else None
            yield SampleInput(input_tensor, n, dim, prepend, append)
    yield SampleInput(make_arg((XS, XS, XS)), S + 1, 1)
    yield SampleInput(make_arg((XS, XS, XS)), S * 3 + 2, 2, make_arg((XS, XS, XS)), make_arg((XS, XS, XS)))

def sample_inputs_histogram(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))
    for (size, bin_ct, weighted, density) in product(sizes, range(1, 5), [False, True], [False, True]):
        input_tensor = make_arg(size)
        weight_tensor = make_arg(size) if weighted else None
        yield SampleInput(input_tensor, bin_ct, weight=weight_tensor, density=density)
        bins_tensor = make_arg((bin_ct + 1,))
        yield SampleInput(input_tensor, bins_tensor, weight=weight_tensor, density=density)

def sample_inputs_histogramdd(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((S, S), (S, S, S), (S, 1, S), (S, 0, S))
    bin_ct_patterns = ((1, 1, 1, 1, 1), (2, 3, 2, 3, 2), (3, 2, 3, 2, 3))
    for (size, bin_ct_pattern, weighted, density) in product(sizes, bin_ct_patterns, [False, True], [False, True]):
        input_tensor = make_arg(size)
        bin_ct = bin_ct_pattern[:size[-1]]
        weight_tensor = make_arg(size[:-1]) if weighted else None
        yield SampleInput(input_tensor, bin_ct, weight=weight_tensor, density=density)
        bins_tensor = [make_arg(ct + 1) for ct in bin_ct]
        yield SampleInput(input_tensor, bins_tensor, weight=weight_tensor, density=density)

def error_inputs_histogramdd(opinfo, device, **kwargs):
    if False:
        print('Hello World!')
    invalid_bins = [1, 1, 1, 1, 1]
    make_arg = partial(make_tensor, dtype=torch.float, device=device, requires_grad=False)
    msg = 'histogramdd: The size of bins must be equal to the innermost dimension of the input.'
    yield ErrorInput(SampleInput(make_arg(5, 6), invalid_bins), error_regex=msg)

def sample_inputs_histc(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))
    for (size, min, max) in product(sizes, [0, -10], [0, 10]):
        yield SampleInput(make_arg(size), min=min, max=max)
        for bins in [1, 3, 10]:
            yield SampleInput(make_arg(size), bins=bins, min=min, max=max)

def sample_inputs_bincount(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for (size, weighted) in product((S, M), [False, True]):
        input_tensor = torch.randint(0, size, (size,), dtype=dtype, device=device)
        weight_tensor = make_arg((size,)) if weighted else None
        max_val = int(input_tensor.max().item())
        for minlength in [0, max_val // 2, max_val, 2 * max_val]:
            yield SampleInput(input_tensor, weights=weight_tensor, minlength=minlength)

def sample_inputs_bucketize(op_info, device, dtype, requires_grad, reference_inputs_mode=False, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = (((), S), ((S,), S), ((S, S), S), ((S, S, S), S), ((S, 1, S), S), ((S, 0, S), S))
    if reference_inputs_mode:
        sizes += (((256,), 128), ((128,), 256), ((32, 32), 11), ((32, 4, 32), 33))
    for ((input_shape, nb), out_int32, right) in product(sizes, [False, True], [False, True]):
        input_tensor = make_arg(input_shape)
        boundaries = make_arg(nb).msort()
        yield SampleInput(input_tensor, boundaries, out_int32=out_int32, right=right)
reference_inputs_bucketize = partial(sample_inputs_bucketize, reference_inputs_mode=True)

def error_inputs_bucketize(opinfo, device, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=torch.float, device=device, requires_grad=False)
    yield ErrorInput(SampleInput(make_arg((S, S, S)), make_arg((S, S))), error_regex='boundaries tensor must be 1 dimension')

def sample_inputs_searchsorted(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = (((0,), ((0,),), False), ((M,), ((), (M,), (M, M)), False), ((0, 0), ((0, 0),), False), ((M, M), ((M, M),), False), ((0, 0, 0), ((0, 0, 0),), False), ((M, M, M), ((M, M, M),), False), ((L,), ((),), True))
    for ((size, input_sizes, is_scalar), noncontiguous, out_int32, right) in product(sizes, [False, True], [False, True], [False, True]):
        unsorted_tensor = make_arg(size, noncontiguous=noncontiguous)
        for input_size in input_sizes:
            input = make_arg(input_size, noncontiguous=noncontiguous)
            if is_scalar:
                input = input.item()
            if np.prod(size) == 0:
                boundary_tensor = unsorted_tensor
                sorter = make_tensor(size, dtype=torch.int64, device=device, noncontiguous=noncontiguous)
            else:
                (boundary_tensor, sorter) = torch.sort(unsorted_tensor)
            side = 'right' if right else 'left'
            yield SampleInput(boundary_tensor, input, out_int32=out_int32, right=right)
            yield SampleInput(boundary_tensor, input, out_int32=out_int32, side=side)
            yield SampleInput(unsorted_tensor, input, out_int32=out_int32, right=right, sorter=sorter)
            yield SampleInput(unsorted_tensor, input, out_int32=out_int32, side=side, sorter=sorter)

def sample_inputs_gradient(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    test_cases_float = (((S,), None, None, 1), ((S,), 2.0, None, 1), ((S, S), None, None, 2), ((S, S), [2.0, 2.1], None, 1), ((S, S), [2.0, 2.1], (0, 1), 1), ((4, 4, 4), [2.0, 1.0], (0, 1), 2))
    for (size, spacing, dim, edge_order) in test_cases_float:
        t = make_arg(size)
        yield SampleInput(t, dim=dim, spacing=spacing, edge_order=edge_order)
    test_cases_tensor = (((3, 3, 3), ((1.1, 2.0, 3.5), (4.0, 2, 6.0)), (0, -1), 1), ((3, 3, 3), ((1.0, 3.0, 2.0), (8.0, 6.0, 1.0)), (0, 1), 2))
    for (size, coordinates, dim, edge_order) in test_cases_tensor:
        t = make_arg(size)
        coordinates_tensor_list = []
        for coords in coordinates:
            a = torch.tensor(coords, device=device)
            coordinates_tensor_list.append(a.to(dtype))
        yield SampleInput(t, dim=dim, spacing=coordinates_tensor_list, edge_order=edge_order)

def sample_inputs_getitem(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    test_args = [([1, 2],), (slice(0, 3),), ([slice(0, 3), 1],), ([[0, 2, 3], [1, 3, 3], [0, 0, 2]],), ([[0, 0, 3], [1, 1, 3], [0, 0, 2]],), ([slice(None), slice(None), [0, 3]],), ([slice(None), [0, 3], slice(None)],), ([[0, 3], slice(None), slice(None)],), ([[0, 3], [1, 2], slice(None)],), ([[0, 3]],), ([[0, 3], slice(None)],), ([[0, 3], Ellipsis],), ([[0, 2, 3], [1, 3, 3], torch.LongTensor([0, 0, 2])],), (index_variable(2, S, device=device),), (mask_not_all_zeros((S,)),)]
    for args in test_args:
        yield SampleInput(make_arg((S, S, S)), args=args)
    yield SampleInput(make_arg((S, S, S, S)), args=([slice(None), [0, 1], slice(None), [0, 1]],))

def sample_inputs_index_put(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for accumulate in [False, True]:
        yield SampleInput(make_arg((S, S)), (index_variable(2, S, device=device),), make_arg((2, S)), accumulate=accumulate)
        mask = torch.zeros(S, dtype=torch.bool) if accumulate else mask_not_all_zeros((S,))
        yield SampleInput(make_arg((S, S)), (mask,), make_arg((S,)), accumulate=accumulate)

def sample_inputs_sort(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10

    def small_3d_unique():
        if False:
            while True:
                i = 10
        res = torch.randperm(S * S * S, dtype=torch.int64, device=device).view(S, S, S)
        res = res.to(dtype).requires_grad_(requires_grad)
        return res

    def large_1d_unique():
        if False:
            while True:
                i = 10
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype).requires_grad_(requires_grad)
        return res
    yield SampleInput(large_1d_unique())
    dims = range(-3, 3)
    flag = [True, False]
    for (dim, descending, stable) in product(dims, flag, flag):
        yield SampleInput(small_3d_unique(), dim, descending)
        if torch.device(device).type == 'cpu':
            yield SampleInput(small_3d_unique(), dim=dim, descending=descending, stable=stable)
    tensor_opt = dict(dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(torch.tensor(1, **tensor_opt))
    yield SampleInput(torch.tensor(1, **tensor_opt), 0)
    yield SampleInput(torch.tensor(1, **tensor_opt), 0, True)
    yield SampleInput(torch.tensor((), **tensor_opt))
    yield SampleInput(torch.tensor((), **tensor_opt), 0)
    yield SampleInput(torch.tensor((), **tensor_opt), 0, True)
    yield SampleInput(small_3d_unique(), stable=True)
    yield SampleInput(small_3d_unique(), dim=0, stable=True)
    yield SampleInput(small_3d_unique(), dim=0, descending=True, stable=True)

def sample_inputs_threshold(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S))
    for x_size in sizes:
        yield SampleInput(make_arg(x_size), make_arg(()).item(), make_arg(()).item())

def sample_inputs_unique(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))
    for (shape, sorted, return_inverse, return_counts, dim) in product(sizes, [False, True], [False, True], [False, True], [None, -2, -1, 0, 1, 2]):
        if 0 in shape and shape.index(0) is not dim:
            continue
        if dim is not None and (dim < -len(shape) or dim >= len(shape)):
            continue
        kwargs = dict(sorted=sorted, return_inverse=return_inverse, return_counts=return_counts, dim=dim)
        input_t = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        yield SampleInput(input_t, **kwargs)
        input_t = make_arg(shape, dtype=torch.bool, requires_grad=False).to(dtype).requires_grad_(requires_grad)
        yield SampleInput(input_t, **kwargs)
        yield SampleInput(make_arg(shape), **kwargs)

def sample_inputs_unique_consecutive(*args, **kwargs):
    if False:
        print('Hello World!')
    for sample_input in sample_inputs_unique(*args, **kwargs):
        if not sample_input.kwargs['sorted']:
            sample_input.kwargs.pop('sorted')
            yield sample_input

def sample_inputs_adaptive_avg_pool1d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((0, 8, 8), (5,)), ((3, 8, 8), 5), ((3, 8, 8), 1))
    for (input_shape, output_size) in cases:
        yield SampleInput(make_arg(input_shape), args=(output_size,))
        yield SampleInput(make_arg(input_shape[1:]), args=(output_size,))

def error_inputs_adaptive_avg_pool1d(opinfo, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg((1, 2, 3)), output_size=()), error_regex="'output_size' should contain one int")
    yield ErrorInput(SampleInput(make_arg((1, 1, 1)), output_size=(-1,)), error_regex='elements of output_size must be greater than or equal to 0')

def sample_inputs_adaptive_avg_pool2d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((1, 8, 8, 8), (5, 7)), ((2, 8, 8, 8), (None, 7)), ((1, 8, 4, 3), (5, None)), ((1, 8, 4, 3), (None, None)), ((1, 8, 4, 3), 5))
    for (input_shape, output_size) in cases:
        yield SampleInput(make_arg(input_shape), args=(output_size,))
        yield SampleInput(make_arg(input_shape[1:]), args=(output_size,))

def error_inputs_adaptive_avg_pool2d(opinfo, device, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg((2, 2)), output_size=(2, 2)), error_type=ValueError, error_regex='Input dimension should be at least 3')
    yield ErrorInput(SampleInput(make_arg((1, 2, 3, 4)), output_size=()), error_regex='output_size must be 2')
    yield ErrorInput(SampleInput(make_arg((1, 1, 1, 1)), output_size=(-1, 0)), error_regex='elements of output_size must be greater than or equal to 0')

def sample_inputs_adaptive_avg_pool3d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((0, 8, 8, 8, 8), (5, 7, 4)), ((1, 8, 4, 3, 7), (None, None, None)), ((1, 8, 4, 3, 7), (1, 1, 1)), ((3, 3, 8, 8, 6), (5, 7, None)), ((1, 3, 8, 8, 6), (5, None, 2)), ((3, 3, 8, 8, 6), (None, 3, 2)))
    for (input_shape, output_size) in cases:
        yield SampleInput(make_arg(input_shape), args=(output_size,))
        yield SampleInput(make_arg(input_shape[1:]), args=(output_size,))

def error_inputs_adaptive_avg_pool3d(opinfo, device, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg((2, 2, 2)), output_size=(2, 2, 2)), error_type=ValueError, error_regex='Input dimension should be at least 4')
    yield ErrorInput(SampleInput(make_arg((1, 2, 3, 4)), output_size=()), error_regex='output_size must be 3')
    yield ErrorInput(SampleInput(make_arg((1, 1, 1, 1, 1)), output_size=(-1, 0, 2)), error_regex='elements of output_size must be greater than or equal to 0')

def sample_inputs_adaptive_max_pool1d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((3, 4, 4), 3), ((3, 4, 4), 1))
    for (shapes, return_idx) in product(cases, (True, False)):
        yield SampleInput(make_arg(shapes[0]), args=(shapes[1], return_idx))
        yield SampleInput(make_arg(shapes[0][1:]), args=(shapes[1], return_idx))

def error_inputs_adaptive_max_pool1d(opinfo, device, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg((1, 2, 3)), output_size=()), error_regex="'output_size' should contain one int")
    yield ErrorInput(SampleInput(make_arg((1, 1, 1)), output_size=(-1,)), error_regex='Trying to create tensor with negative dimension')

def sample_inputs_adaptive_max_pool2d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((1, 4, 4, 4), (2, 3)), ((2, 4, 4, 4), (None, 3)), ((2, 4, 4, 4), (1, 1)), ((1, 4, 4, 3), (3, None)), ((1, 4, 4, 3), (None, None)), ((1, 4, 4, 3), 3))
    for (shapes, return_idx) in product(cases, (True, False)):
        yield SampleInput(make_arg(shapes[0]), args=(shapes[1], return_idx))
        yield SampleInput(make_arg(shapes[0][1:]), args=(shapes[1], return_idx))

def error_inputs_adaptive_max_pool2d(opinfo, device, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg((2, 2)), output_size=(2, 2)), error_type=ValueError, error_regex='Input dimension should be at least 3')
    yield ErrorInput(SampleInput(make_arg((1, 2, 3, 4)), output_size=()), error_regex='internal error')
    yield ErrorInput(SampleInput(make_arg((1, 1, 1, 1)), output_size=(-1, 0)), error_regex='Trying to create tensor with negative dimension')

def sample_inputs_adaptive_max_pool3d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((1, 4, 4, 3, 5), (None, None, None)), ((1, 4, 4, 3, 5), (1, 1, 1)), ((3, 3, 4, 4, 6), (2, 3, None)), ((1, 3, 4, 4, 6), (3, None, 2)), ((3, 3, 4, 4, 6), (None, 3, 2)))
    for (shapes, return_idx) in product(cases, (True, False)):
        yield SampleInput(make_arg(shapes[0]), args=(shapes[1], return_idx))
        yield SampleInput(make_arg(shapes[0][1:]), args=(shapes[1], return_idx))

def error_inputs_adaptive_max_pool3d(opinfo, device, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg((2, 2, 2)), output_size=(2, 2, 2)), error_type=ValueError, error_regex='Input dimension should be at least 4')
    yield ErrorInput(SampleInput(make_arg((1, 2, 3, 4)), output_size=()), error_regex='internal error')
    yield ErrorInput(SampleInput(make_arg((1, 1, 1, 1, 1)), output_size=(-1, 0, 2)), error_regex='Trying to create tensor with negative dimension')

class _TestParamsMaxPoolBase:

    def __init__(self):
        if False:
            return 10
        self.kwargs = {'kernel_size': [3], 'stride': [2, None], 'ceil_mode': [True, False], 'padding': [0, 1], 'dilation': [1], 'return_indices': [True, False]}
        self.shapes = [[1, 2, None], [2], [3, 6]]

    def _gen_shape(self):
        if False:
            return 10
        for shape in product(*self.shapes):
            if shape[0] is None:
                shape = shape[1:]
            yield (shape, torch.contiguous_format)
            if len(self.shapes) == 4 and len(shape) == 4:
                yield (shape, torch.channels_last)

    def _gen_kwargs(self):
        if False:
            while True:
                i = 10
        keys = self.kwargs.keys()
        for values in product(*self.kwargs.values()):
            yield dict(zip(keys, values))

    def gen_input_params(self):
        if False:
            print('Hello World!')
        yield from product(self._gen_shape(), self._gen_kwargs())

class _TestParamsMaxPool1d(_TestParamsMaxPoolBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.kwargs['kernel_size'] += [(3,)]
        self.kwargs['stride'] += [(2,)]
        self.kwargs['padding'] += [(1,)]
        self.kwargs['dilation'] += [(1,)]

class _TestParamsMaxPool2d(_TestParamsMaxPoolBase):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__()
        self.kwargs['kernel_size'] += [(3, 2)]
        self.kwargs['stride'] += [(2, 1)]
        self.kwargs['padding'] += [(1, 1)]
        self.kwargs['dilation'] += [(1, 2)]
        self.shapes.append([6])

class _TestParamsMaxPool3d(_TestParamsMaxPoolBase):

    def __init__(self):
        if False:
            return 10
        super().__init__()
        self.kwargs['kernel_size'] += [(3, 2, 3)]
        self.kwargs['stride'] += [(2, 1, 2)]
        self.kwargs['dilation'] += [(1, 2, 1)]
        self.shapes.append([6])
        self.shapes.append([5])

def sample_inputs_max_pool(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    params_generator_type_dict = {'nn.functional.max_pool1d': _TestParamsMaxPool1d, 'nn.functional.max_pool2d': _TestParamsMaxPool2d, 'nn.functional.max_pool3d': _TestParamsMaxPool3d, 'max_pool2d_with_indices_backward': _TestParamsMaxPool2d}
    params_generator = params_generator_type_dict[op_info.name]()
    for ((shape, memory_format), kwargs) in params_generator.gen_input_params():
        arg = make_arg(shape).to(memory_format=memory_format).requires_grad_(requires_grad)
        yield SampleInput(arg, kwargs=kwargs)

def max_pool2d_backward(*args, kernel_size=(), stride=(), padding=(0,), dilation=(1,), ceil_mode=False, **kwargs):
    if False:
        return 10
    (out, indices) = torch.nn.functional.max_pool2d_with_indices(*args, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, ceil_mode=ceil_mode, return_indices=True)
    grad_out = torch.ones_like(out)
    if stride is None:
        stride = kernel_size
    out_b = torch.ops.aten.max_pool2d_with_indices_backward.default(grad_out, *args, kernel_size, stride, padding, dilation, ceil_mode, indices)
    return out_b

def error_inputs_max_pool1d(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    for requires_grad in (True, False):
        make_arg = partial(make_tensor, device=device, dtype=torch.float, requires_grad=requires_grad)
        x = make_arg((0, 1, 49))
        yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': -1, 'return_indices': True}), error_regex='pad must be non-negative')
        yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 4, 'return_indices': True}), error_regex='pad should be at most half of kernel size')
        error_msg = 'Expected 2D or 3D \\(batch mode\\) tensor with optional 0 dim batch size for input'
        yield ErrorInput(SampleInput(make_arg((), requires_grad=requires_grad), kwargs={'kernel_size': 1}), error_regex=error_msg)
        yield ErrorInput(SampleInput(torch.tensor([], device=device, requires_grad=requires_grad), kwargs={'kernel_size': 1}), error_regex=error_msg)
        yield ErrorInput(SampleInput(make_arg((0, 10), requires_grad=requires_grad), kwargs={'kernel_size': 1}), error_regex=error_msg)
        yield ErrorInput(SampleInput(make_arg((1, 10, 0), requires_grad=requires_grad), kwargs={'kernel_size': 1}), error_regex=error_msg)
        error_msg = 'stride must be greater than zero, but got 0'
        yield ErrorInput(SampleInput(make_arg((3, 3, 3)), kwargs={'kernel_size': 1, 'stride': 0}), error_regex=error_msg)
        error_msg = 'dilation must be greater than zero, but got 0'
        yield ErrorInput(SampleInput(make_arg((3, 3, 3)), kwargs={'kernel_size': 1, 'stride': 1, 'padding': 0, 'dilation': 0}), error_regex=error_msg)
        error_msg = 'Invalid computed output size: -2'
        yield ErrorInput(SampleInput(make_arg((2, 2, 2)), kwargs={'kernel_size': 5, 'stride': 1, 'padding': 0, 'dilation': 1}), error_regex=error_msg)
        error_msg = 'kernel_size must be greater than zero'
        yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 0}), error_regex=error_msg)
        error_msg = 'stride must be greater than zero'
        yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 0}), error_regex=error_msg)

def error_inputs_max_pool2d(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float, requires_grad=False)
    x = make_arg((0, 1, 49))
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': -1, 'return_indices': True}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2), 'stride': 50, 'padding': -1, 'return_indices': True}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 4, 'return_indices': True}), error_regex='pad should be at most half of kernel size')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2), 'stride': 50, 'padding': 4, 'return_indices': True}), error_regex='pad should be at most half of kernel size')
    err_msg = 'Expected 3D or 4D \\(batch mode\\) tensor with optional 0 dim batch size for input'
    yield ErrorInput(SampleInput(make_arg((1, 0, 10)), kwargs={'kernel_size': 1}), error_regex=err_msg)
    yield ErrorInput(SampleInput(make_arg((2, 1, 10, 0)), kwargs={'kernel_size': 1}), error_regex=err_msg)

def error_inputs_max_pool3d(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float, requires_grad=False)
    x = make_arg((0, 1, 49, 50))
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': -1, 'return_indices': True}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2, 2), 'stride': 50, 'padding': -1, 'return_indices': True}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 4, 'return_indices': True}), error_regex='pad should be at most half of kernel size')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2, 2), 'stride': 50, 'padding': 4, 'return_indices': True}), error_regex='pad should be at most half of kernel size')
    err_msg = "Expected input\\'s non-batch dimensions to have positive length"
    yield ErrorInput(SampleInput(make_arg((0, 1, 2, 10)), kwargs={'kernel_size': 1}), error_regex=err_msg)
    yield ErrorInput(SampleInput(make_arg((2, 1, 0, 1, 2)), kwargs={'kernel_size': 1}), error_regex=err_msg)

def sample_inputs_normalize(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, low=-1, high=1, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], dict] = (((2, 1, 4, 5), {'p': 1.0, 'dim': 2}), ((2, 3, 4, 5), {'p': 2.0, 'dim': 1}), ((1, 2, 4, 5), {'p': 0.5, 'dim': 0}), ((1, 3, 4, 5), {'p': -1.0, 'dim': 1}), ((1, 3, 4, 5), {'p': 0.0, 'dim': -1}), ((), {'p': 1.2, 'dim': 0}), ((2, 3, 4, 5), {}), ((2, 3, 4, 5), {'eps': 0.0001}))
    for (input_shape, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), kwargs=kwargs)

def complex_conv(fn, input_size, weight, grad_output, stride, padding, dilation, groups):
    if False:
        while True:
            i = 10
    grad_output_ = torch.view_as_real(grad_output)
    grad_output_r = grad_output_[..., 0]
    grad_output_i = grad_output_[..., 1]
    weight_ = torch.view_as_real(weight)
    weight_r = weight_[..., 0]
    weight_i = weight_[..., 1]
    a = fn(input_size, weight_r, grad_output_r, stride, padding, dilation, groups)
    b = fn(input_size, weight_i, grad_output_i, stride, padding, dilation, groups)
    c = fn(input_size, weight_r + weight_i, grad_output_r + grad_output_i, stride, padding, dilation, groups)
    return a - b + 1j * (c - a - b)

def conv_transpose_ref(input, weight, bias, stride=1, padding=0, output_padding=0, dilation=1, groups=1, fn=None):
    if False:
        i = 10
        return i + 15
    assert fn is not None
    grad_fn_map = {torch.nn.functional.conv_transpose1d: torch.nn.grad.conv1d_input, torch.nn.functional.conv_transpose2d: torch.nn.grad.conv2d_input, torch.nn.functional.conv_transpose3d: torch.nn.grad.conv3d_input}
    batched_dim_map = {torch.nn.functional.conv_transpose1d: 3, torch.nn.functional.conv_transpose2d: 4, torch.nn.functional.conv_transpose3d: 5}
    (input, weight) = (torch.from_numpy(input), torch.from_numpy(weight))
    is_batched = len(input.shape) == batched_dim_map[fn]
    if not is_batched:
        input = input.unsqueeze(0)
    if bias is not None:
        bias = torch.from_numpy(bias)
        unsqueeze_dims = input.ndim - 2
        for _ in range(unsqueeze_dims):
            bias = bias.unsqueeze(1)
    grad_output = input
    conv_transpose_output = fn(grad_output.to('meta'), weight.to('meta'), None, stride=stride, padding=padding, output_padding=output_padding, groups=groups, dilation=dilation)
    input_size = conv_transpose_output.shape
    grad_fn = grad_fn_map[fn]
    if weight.dtype.is_complex:
        out = complex_conv(grad_fn, input_size, weight, grad_output, stride, padding, dilation, groups)
    else:
        out = grad_fn(input_size, weight, grad_output, stride, padding, dilation, groups)
    if bias is not None:
        out = out + bias
    return out.squeeze(0) if not is_batched else out

def sample_inputs_conv_transpose1d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], Tuple[int], Tuple[int], dict] = (((1, 3, 4), (3, 3, 3), (3,), {'stride': (2,), 'padding': 2, 'output_padding': (1,), 'groups': 1}), ((2, 2, 4), (2, 2, 4), (4,), {'stride': (3,), 'padding': (1,), 'output_padding': (2,), 'groups': 2, 'dilation': (4,)}), ((1, 1, 4), (1, 1, 4), (1,), {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1, 'dilation': (2,)}), ((1, 1, 4), (1, 2, 3), None, {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1}), ((1, 4, 5), (4, 8, 3), None, {}))
    for (input_shape, weight, bias, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)
        yield SampleInput(make_arg(input_shape[1:]), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)

def sample_inputs_conv_transpose2d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], Tuple[int], Tuple[int], dict] = (((1, 3, 4, 4), (3, 3, 3, 3), (3,), {'stride': (2, 2), 'padding': 2, 'output_padding': (1, 1), 'groups': 1}), ((2, 2, 4, 4), (2, 2, 4, 5), (4,), {'stride': (3, 2), 'padding': (1, 2), 'output_padding': (2, 3), 'groups': 2, 'dilation': (4, 4)}), ((1, 1, 4, 5), (1, 1, 4, 3), (1,), {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1, 'dilation': (2, 3)}), ((1, 1, 4, 3), (1, 2, 3, 4), None, {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1}), ((2, 4, 4, 4), (4, 1, 3, 3), None, {'groups': 4}), ((1, 2, 5, 5), (2, 4, 3, 3), None, {}))
    for (input_shape, weight, bias, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)
        yield SampleInput(make_arg(input_shape[1:]), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)

def sample_inputs_conv_transpose3d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], Tuple[int], Tuple[int], dict] = (((1, 3, 4, 4, 4), (3, 3, 3, 3, 3), (3,), {'stride': (2, 2, 2), 'padding': 2, 'output_padding': (1, 1, 1), 'groups': 1}), ((2, 2, 4, 4, 4), (2, 2, 4, 5, 6), (4,), {'stride': (3, 2, 1), 'padding': (1, 2, 3), 'output_padding': (2, 3, 1), 'groups': 2, 'dilation': (4, 4, 4)}), ((1, 1, 4, 5, 2), (1, 1, 4, 3, 1), (1,), {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1, 'dilation': (2, 3, 2)}), ((1, 1, 4, 3, 4), (1, 2, 3, 4, 5), None, {'stride': 2, 'padding': 1, 'output_padding': 1, 'groups': 1}), ((1, 4, 5, 5, 5), (4, 8, 3, 3, 3), None, {}))
    for (input_shape, weight, bias, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)
        yield SampleInput(make_arg(input_shape[1:]), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)

def sample_inputs_conv1d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple = (((1, 3, 4), (3, 3, 3), (3,), {'stride': (2,), 'padding': 2, 'groups': 1}), ((2, 4, 8), (2, 2, 3), (2,), {'stride': 3, 'padding': 1, 'groups': 2, 'dilation': 2}), ((1, 4, 5), (1, 4, 3), None, {'stride': (2,), 'padding': 'valid'}), ((2, 2, 4), (2, 1, 4), (2,), {'stride': (1,), 'padding': 'same', 'groups': 2, 'dilation': (2,)}), ((1, 4, 5), (3, 4, 3), None, {}))
    for (input_shape, weight, bias, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)
        yield SampleInput(make_arg(input_shape[1:]), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)

def error_inputs_conv1d(opinfo, device, **kwargs):
    if False:
        while True:
            i = 10
    input = torch.randn(size=(33, 16, 30), device=device, dtype=torch.float64)
    weight = torch.randn(size=(20, 16, 5), device=device, dtype=torch.float64)
    groups = 0
    yield ErrorInput(SampleInput(input, kwargs={'weight': weight, 'groups': groups}), error_regex='non-positive groups is not supported')

def error_inputs_conv2d(opinfo, device, **kwargs):
    if False:
        return 10
    weight = torch.randint(high=10, size=(3, 2, 3, 3), device=device)
    input = torch.randint(high=10, size=(2, 4, 4), device=device)
    bias = torch.rand((3,), dtype=torch.float32, device=device)
    yield ErrorInput(SampleInput(input, args=(weight, bias)), error_regex='should be the same')
    weight = torch.rand(size=(3, 2, 3, 3), device=device, dtype=torch.float64)
    input = torch.rand(size=(2, 4, 4), device=device, dtype=torch.float64)
    bias = torch.rand((3,), dtype=torch.complex128, device=device)
    yield ErrorInput(SampleInput(input, args=(weight, bias)), error_regex='should be the same')
    input = torch.randn(size=(1, 4, 5, 5), device=device, dtype=torch.float64)
    weight = torch.randn(size=(8, 4, 3, 3), device=device, dtype=torch.float64)
    groups = 0
    yield ErrorInput(SampleInput(input, kwargs={'weight': weight, 'groups': groups}), error_regex='non-positive groups is not supported')

def sample_inputs_conv2d(op_info, device, dtype, requires_grad, jit_fail_sample=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple = (((1, 3, 4, 4), (3, 3, 3, 3), (3,), {'stride': (2, 2), 'padding': 2, 'groups': 1}), ((2, 4, 8, 8), (2, 2, 3, 3), (2,), {'stride': (3, 2), 'padding': (2, 1), 'groups': 2, 'dilation': (4, 4)}), ((1, 4, 5, 5), (1, 4, 2, 3), (1,), {'stride': 2, 'padding': 1, 'groups': 1, 'dilation': (2, 3)}), ((1, 4, 5, 5), (1, 4, 2, 3), (1,), {'stride': 2, 'padding': 1, 'groups': 1, 'dilation': (2, 3)}), ((1, 2, 4, 3), (4, 2, 3, 4), None, {'stride': 2, 'padding': 1, 'groups': 1}), ((1, 4, 5, 5), (1, 4, 2, 3), (1,), {'stride': 2, 'padding': 'valid'}), ((1, 4, 5, 5), (1, 4, 2, 3), (1,), {'stride': 1, 'padding': 'same', 'dilation': 3}), ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {'groups': 4}), ((2, 4, 6, 6), (8, 1, 3, 3), (8,), {'groups': 4}), ((2, 4, 6, 6), (8, 1, 3, 3), None, {'groups': 4}), ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {'groups': 4, 'stride': (3, 2)}), ((2, 4, 6, 6), (4, 1, 3, 3), (4,), {'groups': 4, 'padding': (1, 1)}), ((2, 4, 5, 5), (4, 1, 2, 2), (4,), {'groups': 4, 'dilation': (2, 2)}), ((2, 4, 6, 5), (6, 2, 3, 2), (6,), {'groups': 2}), ((1, 4, 5, 5), (3, 4, 3, 3), None, {}))
    for (input_shape, weight, bias, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)
        yield SampleInput(make_arg(input_shape[1:]), args=(make_arg(weight), make_arg(bias) if bias is not None else bias), kwargs=kwargs)

def sample_inputs_group_norm(opinfo, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], int, float] = (((1, 6, 3), 2, {'eps': 0.5}), ((2, 6, 3), 2, {'eps': -0.5}), ((1, 3), 1, {'eps': 1e-05}), ((0, 2), 1, {'eps': 1e-05}), ((S, S, S), 1, {'eps': 0.5}))
    for (input_shape, num_groups, kwargs) in cases:
        channels = input_shape[1] if len(input_shape) > 1 else 0
        weight_tensor = make_arg(channels)
        bias_tensor = make_arg(channels)
        weights = [weight_tensor, None]
        biases = [bias_tensor, None]
        for (weight, bias) in itertools.product(weights, biases):
            kwargs = {'weight': weight, 'bias': bias, **kwargs}
            yield SampleInput(make_arg(input_shape), num_groups, **kwargs)
    yield SampleInput(make_arg((1, 2)), args=(1,))

def reference_inputs_group_norm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_group_norm(op_info, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], int, float] = (((20, 6, 10, 10), 3, {'eps': 1e-05}), ((20, 6, 10, 10), 6, {'eps': 1e-05}), ((20, 6, 10, 10), 1, {'eps': 1e-05}))
    for (input_shape, num_groups, kwargs) in cases:
        channels = input_shape[1] if len(input_shape) > 1 else 0
        input_tensor = make_arg(input_shape)
        weight_tensor = make_arg(channels)
        bias_tensor = make_arg(channels)
        weights = [weight_tensor, None]
        biases = [bias_tensor, None]
        for (weight, bias) in itertools.product(weights, biases):
            kwargs = {'weight': weight, 'bias': bias, **kwargs}
            yield SampleInput(input_tensor, num_groups, **kwargs)

def sample_inputs_instance_norm(opinfo, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_arg_without_requires_grad = partial(make_tensor, device=device, dtype=dtype, requires_grad=False)
    cases: Tuple[Tuple[int], dict] = (((S, S, S), {'momentum': 0.5, 'eps': 0.6}), ((S, S, S), {'momentum': 0.5, 'eps': 0.6, 'use_input_stats': True}), ((3, 2, 4), {'momentum': -1.2}), ((3, 2, 4), {'momentum': 0.0}), ((3, 2, 3, 4), {'momentum': -1.0, 'eps': 0.5}), ((3, 2, 3, 4), {'momentum': -1.0, 'eps': 0.5}))
    for (input_shape, kwargs) in cases:
        channels = input_shape[1]
        weight = make_arg(channels)
        bias = make_arg(channels)
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        new_kwargs = {'running_mean': running_mean, 'running_var': running_var, 'weight': weight, 'bias': bias, **kwargs}
        yield SampleInput(make_arg(input_shape), args=(), kwargs=new_kwargs)
    weights = [channels, None]
    biases = [None, None]
    for (weight_channels, bias_channels) in zip(weights, biases):
        running_mean = make_arg_without_requires_grad(channels, low=0)
        running_var = make_arg_without_requires_grad(channels, low=0)
        yield SampleInput(make_arg(input_shape), args=(), kwargs={'running_mean': running_mean, 'running_var': running_var, 'weight': make_arg(weight_channels) if weight_channels is not None else None, 'bias': make_arg(bias_channels) if bias_channels is not None else None})
    yield SampleInput(make_arg((1, 2, 3)), kwargs={})

def sample_inputs_layer_norm(opinfo, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], Tuple[int], dict] = (((1, 2, 3), (1, 2, 3), {'eps': 0.5}), ((2, 2, 3), (2, 3), {'eps': -0.5}), ((1,), (1,), {}), ((1, 2), (2,), {}), ((0, 1), (1,), {}))
    for (input_shape, normalized_shape, kwargs) in cases:
        weight = make_arg(normalized_shape)
        bias = make_arg(normalized_shape)
        yield SampleInput(make_arg(input_shape), args=(normalized_shape, weight, bias), kwargs=kwargs)
    yield SampleInput(make_arg((1, 2)), args=((2,),))

def sample_inputs_native_layer_norm(opinfo, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], Tuple[int], float] = (((1, 2, 3), (1, 2, 3), 0.5), ((2, 2, 3), (2, 3), -0.5), ((1,), (1,), 1e-05), ((1, 2), (2,), 1e-05), ((0, 1), (1,), 1e-05))
    for (input_shape, normalized_shape, eps) in cases:
        weight = make_arg(normalized_shape)
        bias = make_arg(normalized_shape)
        yield SampleInput(make_arg(input_shape), args=(normalized_shape, weight, bias, eps))
        yield SampleInput(make_arg(input_shape), args=(normalized_shape, None, bias, eps))
        yield SampleInput(make_arg(input_shape), args=(normalized_shape, weight, None, eps))
        yield SampleInput(make_arg(input_shape), args=(normalized_shape, None, None, eps))

def error_inputs_group_norm(opinfo, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=torch.float32, requires_grad=False)
    err_msg1 = 'Expected at least 2 dimensions for input tensor but received'
    s1 = SampleInput(make_arg(1), args=(1,))
    yield ErrorInput(s1, error_regex=err_msg1)
    err_msg2 = 'Expected number of channels in input to be divisible by num_groups, but got input of shape'
    s2 = SampleInput(make_arg((2, 7, 4)), args=(2,))
    yield ErrorInput(s2, error_regex=err_msg2)

def error_inputs_native_layer_norm(opinfo, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32, requires_grad=False)
    input_shape = (1, 2, 3)
    err_msg1 = 'Expected normalized_shape to be at least 1-dimensional'
    s1 = SampleInput(make_arg(input_shape), args=(tuple(), None, None, 1e-05))
    yield ErrorInput(s1, error_regex=err_msg1)
    normalized_shape = (1, 2, 3)
    weight = make_arg((1, 2))
    err_msg2 = 'Expected weight to be of same shape as normalized_shape'
    s2 = SampleInput(make_arg(input_shape), args=(normalized_shape, weight, None, 1e-05))
    yield ErrorInput(s2, error_regex=err_msg2)
    bias = make_arg((1, 2))
    err_msg3 = 'Expected bias to be of same shape as normalized_shape'
    s3 = SampleInput(make_arg(input_shape), args=(normalized_shape, None, bias, 1e-05))
    yield ErrorInput(s3, error_regex=err_msg3)
    err_msg4 = 'Given normalized_shape='
    s4 = SampleInput(make_arg((2, 2, 3)), args=((2, 2), None, None, 1e-05))
    yield ErrorInput(s4, error_regex=err_msg4)

def sample_inputs_local_response_norm(opinfo, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: Tuple[Tuple[int], Tuple[int], dict] = (((1, 6, 3), 2, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}), ((1, 6, 3), 2, {'beta': 0.5, 'k': 1.25}), ((1, 6, 3), 2, {'alpha': 3e-05, 'k': 1.25}), ((1, 6, 3), 2, {'alpha': 3e-05, 'beta': 0.5}), ((1, 6, 3), 2, {'alpha': 3e-05}), ((1, 6, 3), 2, {'beta': 0.5}), ((1, 6, 3), 2, {'k': 1.25}), ((1, 6, 3), 2, {}), ((2, 6, 3), 2, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}), ((1, 1, 2), 1, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}), ((0, 1, 2), 1, {'alpha': 3e-05, 'beta': 0.5, 'k': 1.25}))
    for (input_shape, size, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(size,), kwargs=kwargs)

def sample_inputs_hardswish(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    N = 5
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-5, high=5)
    return (SampleInput(make_arg((N * 2, N * 2))) for _ in range(1, N))

def sample_inputs_linear(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    features_options = [[3, 4], [8, 8]]
    batch_options: List[List[int]] = [[], [0], [8], [2, 3]]
    create_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-2, high=2)
    for (has_bias, (in_feat, out_feat), batch_shape) in itertools.product([True, False], features_options, batch_options):
        input_tensor = create_tensor(batch_shape + [in_feat])
        weight = create_tensor([out_feat, in_feat])
        if not has_bias:
            yield SampleInput(input_tensor, weight)
            continue
        bias = create_tensor([out_feat])
        yield SampleInput(input_tensor, weight, bias)

def sample_inputs_bilinear(self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    features_options = [[3, 4, 5], [8, 8, 8]]
    batch_options: List[List[int]] = [[], [0], [8], [2, 3]]
    create_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-2, high=2)
    for (has_bias, (in_feat1, in_feat2, out_feat), batch_shape) in itertools.product([True, False], features_options, batch_options):
        input_tensor1 = create_tensor(batch_shape + [in_feat1])
        input_tensor2 = create_tensor(batch_shape + [in_feat2])
        weight = create_tensor([out_feat, in_feat1, in_feat2])
        if not has_bias:
            yield SampleInput(input_tensor1, input_tensor2, weight)
            continue
        bias = create_tensor([out_feat])
        yield SampleInput(input_tensor1, input_tensor2, weight, bias)

def sample_inputs_glu(self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    features_options = [[2], [2, 4], [8, 8], [3, 6, 8], [1, 4, 6, 7]]
    batch_options: List[List[int]] = [[], [0], [8], [2, 3]]
    create_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-2, high=2)
    for (features, batch_shape) in itertools.product(features_options, batch_options):
        ndim = len(features) + len(batch_shape)
        for dim in range(ndim):
            input_tensor = create_tensor(batch_shape + features)
            dim_size = input_tensor.size(dim)
            if dim_size > 0 and dim_size % 2 == 0:
                yield SampleInput(input_tensor, dim)

def sample_inputs_interpolate(mode, self, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    (N, C) = (2, 3)
    D = 4
    S = 3
    L = 5
    align_corners_options: Tuple[Any, ...] = (None,)
    if mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        align_corners_options = (True, False, None)
    ranks_for_mode = {'nearest': [1, 2, 3], 'nearest-exact': [1, 2, 3], 'linear': [1], 'bilinear': [2], 'bicubic': [2], 'trilinear': [3], 'area': [1, 2, 3]}

    def shape(size, rank, with_batch_channel=True):
        if False:
            print('Hello World!')
        if with_batch_channel:
            return tuple([N, C] + [size] * rank)
        return tuple([size] * rank)
    if mode in ('bilinear', 'bicubic') and dtype == torch.uint8:
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, high=256 if dtype == torch.uint8 else None)
        rank = 2
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            yield SampleInput(make_arg(shape(270, rank), memory_format=memory_format), shape(130, rank, False), scale_factor=None, mode=mode, align_corners=False)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for align_corners in align_corners_options:
        for rank in ranks_for_mode[mode]:
            yield SampleInput(make_arg(shape(D, rank)), shape(S, rank, False), scale_factor=None, mode=mode, align_corners=align_corners)
            yield SampleInput(make_arg(shape(D, rank)), shape(L, rank, False), scale_factor=None, mode=mode, align_corners=align_corners)
            for recompute_scale_factor in [False, True]:
                for scale_factor in [1.7, 0.6]:
                    yield SampleInput(make_arg(shape(D, rank)), size=None, scale_factor=scale_factor, mode=mode, align_corners=align_corners, recompute_scale_factor=recompute_scale_factor)

def reference_inputs_interpolate(mode, self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_interpolate(mode, self, device, dtype, requires_grad, **kwargs)
    if mode in ('bilinear', 'bicubic'):
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, high=256 if dtype == torch.uint8 else None)
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            for aa in [True, False]:
                yield SampleInput(make_arg((2, 3, 345, 456), memory_format=memory_format), (270, 270), scale_factor=None, mode=mode, align_corners=False, antialias=aa)

def sample_inputs_upsample(mode, self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    (N, C) = (2, 3)
    D = 4
    S = 3
    L = 5
    ranks_for_mode = {'nearest': [1, 2, 3], 'bilinear': [2]}

    def shape(size, rank, with_batch_channel=True):
        if False:
            print('Hello World!')
        if with_batch_channel:
            return torch.Size([N, C] + [size] * rank)
        return torch.Size([size] * rank)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for rank in ranks_for_mode[mode]:
        yield SampleInput(make_arg(shape(D, rank)), size=shape(S, rank, False))
        yield SampleInput(make_arg(shape(D, rank)), size=shape(L, rank, False))
        yield SampleInput(make_arg(shape(D, rank)), scale_factor=1.7)
        yield SampleInput(make_arg(shape(D, rank)), scale_factor=0.6)

def reference_inputs_upsample(mode, self, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_upsample(mode, self, device, dtype, requires_grad, **kwargs)
    if mode in ('bilinear',):
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, high=256 if dtype == torch.uint8 else None)
        for memory_format in [torch.contiguous_format, torch.channels_last]:
            yield SampleInput(make_arg((2, 3, 345, 456), memory_format=memory_format), (270, 270))

def sample_inputs_upsample_aa(mode, self, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    N = 6
    C = 3
    H = 10
    W = 20
    S = 3
    L = 5
    input_tensor = make_tensor(torch.Size([N, C, H, W]), device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=False, scale_factors=None)
    yield SampleInput(input_tensor, output_size=torch.Size([L, L]), align_corners=False, scale_factors=None)
    yield SampleInput(input_tensor, output_size=None, align_corners=False, scale_factors=[1.7, 0.9])
    yield SampleInput(input_tensor, output_size=None, align_corners=True, scale_factors=[0.8, 1.0])
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=False, scales_h=None, scales_w=None)
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=False, scales_h=1.7, scales_w=0.9)
    yield SampleInput(input_tensor, output_size=torch.Size([S, S]), align_corners=True, scales_h=1.7, scales_w=0.9)

def sample_inputs_gelu(self, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    N = 5
    for _ in range(1, N):
        for approximate in ['none', 'tanh']:
            yield SampleInput(make_tensor((N * 2, N * 2), device=device, dtype=dtype, requires_grad=requires_grad, low=-3, high=3), approximate=approximate)

def error_inputs_gelu(op, device, **kwargs):
    if False:
        while True:
            i = 10
    yield ErrorInput(SampleInput(make_tensor((), dtype=torch.float, device=device), kwargs={'approximate': 'asdf'}), error_regex='approximate argument must be either')

def sample_inputs_max_min_reduction_with_dim(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    inputs = []
    args_for_reduction_with_dim = (((S, S, S), (1,)), ((S, S, S), (1, True)), ((), (0,)), ((), (0, True)))
    return (SampleInput(make_tensor(input_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad), *args) for (input_tensor, args) in args_for_reduction_with_dim)

def sample_inputs_max_min_reduction_no_dim(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    yield SampleInput(make_arg((S, S, S)))
    yield SampleInput(make_arg(()))

def _generate_nan_reduction_inputs(device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    yield from _generate_reduction_inputs(device, dtype, requires_grad)
    if dtype.is_complex or dtype.is_floating_point:
        yield torch.tensor([2, torch.nan, -1], device=device, dtype=dtype, requires_grad=requires_grad)
        yield torch.tensor([[torch.nan, 2], [0, 1]], device=device, dtype=dtype, requires_grad=requires_grad)

def sample_inputs_nan_reduction(supports_multiple_dims):
    if False:
        print('Hello World!')

    def fn(op_info, device, dtype, requires_grad, **kwargs):
        if False:
            return 10
        for t in _generate_nan_reduction_inputs(device, dtype, requires_grad):
            yield SampleInput(t.clone().requires_grad_(requires_grad))
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims):
                yield SampleInput(t.clone().requires_grad_(requires_grad), **kwargs)
    return fn

def sample_inputs_reduction_quantile(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    test_quantiles = (0.5, make_tensor((2,), dtype=dtype, device=device, low=0, high=1, requires_grad=requires_grad))
    test_interpolations = ['linear', 'midpoint']
    for quantiles in test_quantiles:
        for t in _generate_reduction_inputs(device, dtype, requires_grad):
            input = t.clone().requires_grad_(requires_grad)
            yield SampleInput(input, quantiles)
            for kwargs in _generate_reduction_kwargs(t.ndim, supports_multiple_dims=False):
                kwargs.setdefault('dim', 0)
                kwargs.setdefault('keepdim', False)
                for interpolation in test_interpolations:
                    kwargs['interpolation'] = interpolation
                    input = t.clone().requires_grad_(requires_grad)
                    yield SampleInput(input, quantiles, **kwargs)

def sample_inputs_reduction_count_nonzero(*args, **kwargs):
    if False:
        print('Hello World!')
    'Sample inputs for count_nonzero'
    for sample in sample_inputs_reduction(*args, **kwargs):
        sample.kwargs.pop('keepdim', None)
        yield sample

def sample_inputs_leaky_relu(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    N = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    return (SampleInput(make_arg((N, N))) for _ in range(1, N))

def sample_inputs_fractional_max_pool2d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((1, 3, 9, 9), 3), ((1, 3, 9, 9), (4, 4)), ((1, 3, 9, 9), (6, 6)), ((2, 3, 9, 9), (3, 3)), ((1, 1, 4, 4), (2, 2)), ((1, 2, 6, 6), (4, 4)))
    for (input_shape, kernel_size) in cases:
        for return_indices in [False, True]:
            yield SampleInput(make_arg(input_shape), kernel_size, output_size=2, return_indices=return_indices)
            yield SampleInput(make_arg(input_shape), kernel_size, output_size=(2, 3), return_indices=return_indices)
            yield SampleInput(make_arg(input_shape), kernel_size, output_ratio=(0.5, 0.5), return_indices=return_indices)

def sample_inputs_fractional_max_pool3d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((2, 3, 5, 5, 5), (2, 2, 2)), ((1, 2, 6, 5, 4), 2), ((1, 2, 5, 6, 5), (2, 3, 2)), ((1, 2, 6, 6, 6), (2, 3, 2)), ((1, 1, 7, 6, 7), (2, 3, 4)), ((1, 1, 4, 5, 4), (2, 2, 1)), ((1, 1, 8, 7, 6), (4, 3, 2)), ((0, 1, 4, 5, 4), (2, 2, 1)))
    for (input_shape, kernel_size) in cases:
        for return_indices in [False, True]:
            yield SampleInput(make_arg(input_shape), kernel_size, output_size=2, return_indices=return_indices)
            yield SampleInput(make_arg(input_shape), kernel_size, output_size=(2, 3, 2), return_indices=return_indices)
            yield SampleInput(make_arg(input_shape), kernel_size, output_ratio=(0.5, 0.5, 0.5), return_indices=return_indices)

def sample_inputs_avgpool2d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((1, 3, 9, 9), 3, 1, 1, True, False, 2), ((1, 3, 9, 9), (4, 4), (2, 3), 1, True, False, 2), ((1, 3, 9, 9), (6, 6), (3, 3), (2, 3), True, True, 2), ((2, 3, 9, 9), (3, 3), (1, 1), (1,), True, False, 2), ((1, 1, 4, 4), (2, 2), (), (0,), False, True, -2), ((1, 2, 6, 6), (4, 4), (2, 2), (2,), True, True, None))
    for (input_shape, kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override) in cases:
        yield SampleInput(make_arg(input_shape), args=(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override))
    yield SampleInput(make_arg((1, 3, 9, 9)), args=(3, 3))

def sample_inputs_avgpool1d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: List[Tuple[Tuple[int, ...], Union[int, Tuple[int, ...]], Dict]] = [((2, 3, 9), (3,), {}), ((1, 3, 9), 3, dict(stride=1, padding=1, ceil_mode=True, count_include_pad=False)), ((1, 3, 9), (6,), dict(stride=(3,), padding=(2,), ceil_mode=True, count_include_pad=True)), ((2, 3, 9), (3,), dict(stride=(1,), padding=(1,), ceil_mode=False, count_include_pad=True)), ((0, 3, 9), (6,), dict(stride=(3,), padding=(2,), ceil_mode=False, count_include_pad=True)), ((1, 2, 9), (7,), dict(stride=(3,), padding=(2,), ceil_mode=False)), ((1, 2, 9), (7,), dict(stride=(3,), padding=(3,), ceil_mode=True)), ((1, 2, 9), (7,), dict(stride=(3,), ceil_mode=False)), ((1, 2, 9), (7,), dict(stride=(3,), ceil_mode=True))]
    for (input_shape, kernel_size, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(kernel_size,), kwargs=kwargs)

def sample_inputs_avgpool3d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases: List[Tuple[Tuple[int, ...], Union[int, Tuple[int, ...]], Dict]] = [((2, 3, 3, 4, 4), (2, 2, 2), {}), ((1, 2, 4, 4, 4), 2, dict(stride=1, padding=1, ceil_mode=True, count_include_pad=False, divisor_override=2)), ((1, 2, 5, 5, 5), (2, 3, 4), dict(stride=(1, 2, 2), padding=(0, 1, 2), ceil_mode=True, count_include_pad=True, divisor_override=2)), ((1, 2, 5, 5, 5), (2, 3, 4), dict(stride=(1, 2, 2), padding=(0, 1, 2), ceil_mode=False)), ((1, 1, 7, 5, 7), (6, 3, 4), dict(stride=(2, 3, 2), padding=(3, 1, 0), ceil_mode=False, count_include_pad=False, divisor_override=2)), ((1, 1, 4, 5, 4), (2, 2, 3), dict(stride=(2, 2, 1), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=-2)), ((1, 1, 6, 5, 6), (4, 5, 6), dict(stride=(2, 3, 2), padding=2, ceil_mode=True, count_include_pad=True, divisor_override=None)), ((0, 1, 4, 5, 4), (2, 3, 1), dict(stride=(2, 1, 2), padding=0, ceil_mode=False, count_include_pad=True, divisor_override=None))]
    for (input_shape, kernel_size, kwargs) in cases:
        yield SampleInput(make_arg(input_shape), args=(kernel_size,), kwargs=kwargs)

def error_inputs_avg_pool1d(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    x = torch.rand([0, 1, 49], dtype=torch.float32)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': -1}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 4}), error_regex='pad should be at most half of kernel size')

def error_inputs_avg_pool2d(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    x = torch.rand([0, 1, 49], dtype=torch.float32)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': -1}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2), 'stride': 50, 'padding': -1}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 4}), error_regex='pad should be at most half of kernel size')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2), 'stride': 50, 'padding': 4}), error_regex='pad should be at most half of kernel size')
    x = torch.zeros(3, 3, 3)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (2, 2), 'divisor_override': 0}), error_regex='divisor must be not zero')

def error_inputs_avg_pool3d(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    x = torch.rand([0, 1, 49, 50], dtype=torch.float32)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': -1}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2, 2), 'stride': 50, 'padding': -1}), error_regex='pad must be non-negative')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 4}), error_regex='pad should be at most half of kernel size')
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (3, 2, 2), 'stride': 50, 'padding': 4}), error_regex='pad should be at most half of kernel size')
    x = torch.zeros(3, 3, 3, 3)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': (2, 2, 2), 'divisor_override': 0}), error_regex='divisor must be not zero')
    x = torch.rand([0, 1, 49], dtype=torch.float32)
    yield ErrorInput(SampleInput(x, kwargs={'kernel_size': 2, 'stride': 50, 'padding': 0}), error_regex='non-empty 4D or 5D')

def sample_inputs_to(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    devices = [device]
    if torch.device(device).type == 'cpu':
        devices = [torch.device('cpu'), torch.device('cuda:0')] if torch.cuda.is_available() else devices
    memory_formats = [torch.preserve_format, torch.channels_last]
    for (device, nb, cp, mem_f) in product(devices, [True, False], [True, False], memory_formats):
        kwargs = {'memory_format': mem_f}
        yield SampleInput(make_arg((S, S, S, S)), args=(device, torch.float64, nb, cp), kwargs=kwargs)
    for (nb, cp, mem_f) in product([True, False], [True, False], memory_formats):
        kwargs = {'memory_format': mem_f}
        yield SampleInput(make_arg((S, S, S, S)), args=(torch.float64, nb, cp), kwargs=kwargs)
    for (device, nb, cp, mem_f) in product(devices, [True, False], [True, False], memory_formats):
        kwargs = {'memory_format': mem_f}
        other = make_arg((S, S, S, S), dtype=torch.float64, device=device)
        yield SampleInput(make_arg((S, S, S, S)), args=(other, nb, cp), kwargs=kwargs)

def sample_inputs_topk(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    def get_tensor_input(size):
        if False:
            for i in range(10):
                print('nop')
        return make_tensor(size, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(get_tensor_input((S, M, S)), 3)
    yield SampleInput(get_tensor_input((S, M, S)), 3, 1)
    yield SampleInput(get_tensor_input((S, M, S)), 3, -2)
    yield SampleInput(get_tensor_input((S, M, S)), 3, 1, True)
    yield SampleInput(get_tensor_input((S, M, S)), 3, -2, True)
    yield SampleInput(get_tensor_input((S, M, S)), 3, 1, True, True)
    yield SampleInput(get_tensor_input((S, M, S)), 3, -2, True, True)
    yield SampleInput(get_tensor_input(()), 1)
    yield SampleInput(get_tensor_input(()), 1, 0)
    yield SampleInput(get_tensor_input(()), 1, -1)
    yield SampleInput(get_tensor_input(()), 1, 0, True)
    yield SampleInput(get_tensor_input(()), 1, -1, True)
    yield SampleInput(get_tensor_input(()), 1, 0, True, True)
    yield SampleInput(get_tensor_input(()), 1, -1, True, True)

def sample_inputs_outer(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg(S), make_arg(M))

def sample_inputs_dist(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((S, S, S), (S,), (S, 1, S), (), (S, S))
    ps = (2, 4)
    for (size_x, size_y, p) in product(sizes, sizes, ps):
        yield SampleInput(make_arg(size_x), args=(make_arg(size_y), p))

def sample_inputs_index(op_info, device, dtype, requires_grad, reference=False, **kwargs):
    if False:
        return 10
    select = 'index_select' in op_info.name
    add = 'index_add' in op_info.name
    copy = 'index_copy' in op_info.name
    fill = 'index_fill' in op_info.name
    if reference:
        make_arg = partial(torch.ones, device=device, dtype=dtype, requires_grad=requires_grad)
        make_idx = partial(torch.zeros, device=device, dtype=torch.int64)
    else:
        make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
        if copy or add:
            make_idx = partial(torch.randperm, device=device, dtype=torch.int64)
        else:

            def make_idx(n):
                if False:
                    for i in range(10):
                        print('nop')
                return make_tensor((n,), device=device, dtype=torch.int64, low=0, high=n)
    shapes = [(), (1,), (S, S)]
    if add:
        if dtype == torch.bool:
            alphas = (True, False)
        else:
            alphas = (-1, 0, 2)
    else:
        alphas = (None,)
    if fill:
        values = (make_arg((1,)).item(), make_arg(()))
    else:
        values = (None,)
    for (shape, alpha, value) in product(shapes, alphas, values):
        t = make_arg(shape)
        args = []
        dim = -1 if t.ndim == 2 else 0
        args.append(dim)
        idx = make_idx(t.shape[dim] if t.ndim != 0 else 1)
        args.append(idx)
        if copy or add:
            args.append(make_arg(shape))
        elif fill:
            args.append(value)
        args = tuple(args)
        kwargs = {} if alpha is None else {'alpha': alpha}
        yield SampleInput(t, args=args, kwargs=kwargs)

def sample_inputs_index_reduce(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_idx(n, m):
        if False:
            while True:
                i = 10
        return make_tensor((n,), device=device, dtype=torch.int64, low=0, high=m)
    shapes = [((), ()), ((1,), (1,)), ((S, S), (S, M)), ((S, S, S), (S, M, S))]
    include_selfs = (True, False)
    reduces = ('prod', 'mean', 'amin', 'amax')
    for (shape, include_self, reduce) in product(shapes, include_selfs, reduces):
        (self_shape, src_shape) = shape
        dim = 1 if len(self_shape) >= 2 else 0
        idx = make_idx(src_shape[dim] if len(src_shape) != 0 else 1, self_shape[dim] if len(self_shape) != 0 else 1)
        args = (dim, idx, make_arg(src_shape), reduce)
        yield SampleInput(make_arg(self_shape), args=args, kwargs={'include_self': include_self})
    if requires_grad:
        input = torch.tensor([[0, 13], [0, 0], [15, 19]], dtype=dtype, device=device, requires_grad=requires_grad)
        src = torch.tensor([[2, 0], [0, 0], [2, 3], [2, 2]], dtype=dtype, device=device, requires_grad=requires_grad)
        idx = torch.tensor([0, 1, 2, 0], dtype=torch.long, device=device)
        yield SampleInput(input, args=(0, idx, src, 'prod'), kwargs={'include_self': True})

def sample_inputs_mode(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    args = (((S, S, S), ()), ((S, S, S), (1,)), ((S, S, S), (1, True)), ((), ()), ((), (0,)), ((), (0, True)), ((3000,), ()))
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad, low=None, high=None)
    return (SampleInput(make_arg(input_tensor), *args) for (input_tensor, args) in args)

def sample_inputs_put(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)
    S = 3
    idx = torch.randperm(S * S, device=device, dtype=torch.int64)[:S]
    idx_list = [idx, -idx - 1]
    for (idx, acc) in product(idx_list, (True, False)):
        yield SampleInput(input=make_arg((S, S)), args=(idx.clone(), make_arg((S,)), acc))
    scalar_sizes = [(), (1,)]
    tgt_gen = (make_arg(size) for size in scalar_sizes)
    idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
    src_gen = (make_arg(size) for size in scalar_sizes)
    for (tgt, idx, src, acc) in product(tgt_gen, idx_gen, src_gen, (True, False)):
        yield SampleInput(input=tgt.clone().requires_grad_(requires_grad), args=(idx.clone(), src.clone().requires_grad_(requires_grad), acc))
    tgt_sizes = [(0,), (), (1,), (3, 2)]
    tgt_gen = (make_arg(size) for size in tgt_sizes)
    idx = make_idx((0,), high=1)
    src = make_arg((0,))
    for (tgt, acc) in product(tgt_gen, (True, False)):
        yield SampleInput(input=tgt.clone().requires_grad_(requires_grad), args=(idx.clone(), src.clone().requires_grad_(requires_grad), acc))

def sample_inputs_take(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    make_idx = partial(make_tensor, low=0, dtype=torch.int64, device=device, requires_grad=False)
    S = 3
    index = make_idx((S,), high=S * S)
    for idx in (index, -index - 1):
        yield SampleInput(input=make_arg((S, S)), args=(idx,))
    scalar_sizes = [(), (1,)]
    src_gen = (make_arg(size) for size in scalar_sizes)
    idx_gen = (make_idx(size, high=1) for size in scalar_sizes)
    for (src, idx) in product(src_gen, idx_gen):
        yield SampleInput(input=src.clone().requires_grad_(requires_grad), args=(idx.clone(),))
    src_sizes = [(0,), (), (1,), (3, 2)]
    src_gen = (make_arg(size) for size in src_sizes)
    idx = make_idx((0,), high=1)
    for src in src_gen:
        yield SampleInput(input=src.clone().requires_grad_(requires_grad), args=(idx.clone(),))

def sample_movedim_moveaxis(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg((4, 3, 2, 1)), [0, 1, 2, 3], [3, 2, 1, 0])
    yield SampleInput(make_arg((4, 3, 2, 1)), [0, -1, -2, -3], [-3, -2, -1, -0])

def reference_movedim_moveaxis(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_movedim_moveaxis(op_info, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    args = (((), (), ()), ((3, 5, 7, 2), -2, 1), ((3, 5, 7, 2), (-1, 0), (0, -1)), ((2, 3, 4, 5, 6), (3, -3, 4), (1, 0, -1)), ((2, 3, 4, 5, 6), (-3, 4, 3, 1), (-3, 4, 3, 1)), ((6, 2, 3, 5, 4), (4, 3, 2, 1, 0), (0, 1, 2, 3, 4)), ((6, 2, 3, 5, 4), (-3, -2, -4, -5, -1), (2, 1, 3, 4, 0)), ((6, 2, 3, 5, 4), (4, -2, 2, -4, -5), (-5, 1, 2, -2, -1)))
    for (shape, source, destination) in args:
        yield SampleInput(make_arg(shape), args=(source, destination))

def error_movedim_moveaxis(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((3, -3), (1, 0, -1))), error_regex='movedim: Invalid source or destination dims: source \\(\\[3, -3\\] dims\\) should contain the same number of dims as destination \\(\\[1, 0, -1\\] dims\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((3, -3, 4), (1, 0))), error_regex='movedim: Invalid source or destination dims: source \\(\\[3, -3, 4\\] dims\\) should contain the same number of dims as destination \\(\\[1, 0\\] dims\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((0, 4, -5), (1, 0, 2))), error_regex='movedim: repeated dim in `source` \\(\\[0, 4, -5\\]\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((1, 0, 2), (0, 4, -5))), error_regex='movedim: repeated dim in `destination` \\(\\[0, 4, -5\\]\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((1, 0, -4), (0, 4, -5))), error_regex='movedim: repeated dim in `source` \\(\\[1, 0, -4\\]\\)')
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((0, 1, -6), (1, 4, 2))), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=((1, 4, 2), (0, 1, -6))), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=(-6, 1)), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)
    yield ErrorInput(SampleInput(make_arg(2, 3, 4, 5, 6), args=(3, -6)), error_regex='Dimension out of range \\(expected to be in range of \\[-5, 4\\], but got -6\\)', error_type=IndexError)

def sample_repeat_tile(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    rep_dims = ((), (0,), (1,), (0, 2), (1, 1), (2, 3), (2, 3, 2), (0, 2, 3), (2, 1, 1, 1))
    shapes = ((), (0,), (2,), (3, 0), (3, 2), (3, 0, 1))
    if requires_grad:
        rep_dims = ((), (0,), (0, 2), (1, 1), (2, 3), (1, 3, 2), (3, 1, 1))
        shapes = ((), (0,), (2,), (3, 2))
    is_repeat_op = op_info.name in ['repeat', '_refs.repeat']
    for (rep_dim, shape) in product(rep_dims, shapes):
        if is_repeat_op and len(rep_dim) < len(shape):
            continue
        yield SampleInput(make_arg(shape), rep_dim)

def sample_inputs_narrow_narrow_copy(op_info, device, dtype, requires_grad, *, is_narrow, **kwargs):
    if False:
        while True:
            i = 10
    shapes_and_args = (((S, S, S), 1, 2, 2), ((S, S, S), -1, 2, 2), ((S, S, S), 1, 0, 0), ((S, S, S), -1, 0, 0), ((S, S, S), 2, 1, 2))
    for (shape, dim, start, length) in shapes_and_args:
        tensor = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(tensor, dim, start, length)
        if is_narrow:
            yield SampleInput(tensor, dim, torch.tensor(start), length)

def reference_inputs_narrow_narrow_copy(op_info, device, dtype, requires_grad, *, is_narrow, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    yield from sample_inputs_narrow_narrow_copy(op_info, device, dtype, requires_grad, is_narrow=is_narrow, **kwargs)
    shapes_and_args = (((M,), 0, 0, 0), ((M,), -1, -1, 0), ((M,), 0, 5, 3), ((M,), 0, -5, 2), ((M,), -1, 0, M), ((M,), 0, -M, M), ((M, S), 1, 0, 0), ((S, M), -2, -1, 0), ((L, S), 1, 2, 3), ((L, S), -1, 3, 2), ((M, L), 0, 0, M), ((M, L), -1, -L, L), ((L, M, S), 2, 0, 0), ((M, S, L), -1, -1, 0), ((S, L, M), 2, 0, M), ((L, S, M), -1, -M, M), ((S, L, M), 1, 0, 0), ((S, L, M), 0, 2, 1), ((M, S, M), -1, -5, 4))
    for (shape, dim, start, length) in shapes_and_args:
        tensor = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(tensor, dim, start, length)
        if is_narrow:
            yield SampleInput(tensor, dim, torch.tensor(start), length)

def error_inputs_narrow_narrow_copy(op_info, device, *, is_narrow, is_ref):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg(()), 0, 0, 1), error_type=RuntimeError, error_regex='narrow\\(\\) cannot be applied to a 0-dim tensor\\.')
    if not is_narrow and (not is_ref) and (torch.device(device).type == 'cpu'):
        yield ErrorInput(SampleInput(make_arg((M, S, L)), 3, 0, 0), error_type=RuntimeError, error_regex='Expected dim < static_cast<int64_t>\\(self_sizes.size\\(\\)\\) to be true, but got false\\.')
    else:
        yield ErrorInput(SampleInput(make_arg((M, S, L)), 3, 0, 0), error_type=IndexError, error_regex='Dimension out of range \\(expected to be in range of \\[-3, 2\\], but got 3\\)')
    yield ErrorInput(SampleInput(make_arg((L, S, M)), -4, 0, 0), error_type=IndexError, error_regex='Dimension out of range \\(expected to be in range of \\[-3, 2\\], but got -4\\)')
    yield ErrorInput(SampleInput(make_arg((L, M, S)), 1, M + 1, 0), error_type=IndexError, error_regex='start out of range \\(expected to be in range of \\[-10, 10\\], but got 11\\)')
    yield ErrorInput(SampleInput(make_arg((L, M, S)), 1, -M - 1, 0), error_type=IndexError, error_regex='start out of range \\(expected to be in range of \\[-10, 10\\], but got -11\\)')
    yield ErrorInput(SampleInput(make_arg((S, L, M)), 2, 0, M + 1), error_type=RuntimeError, error_regex='start \\(0\\) \\+ length \\(11\\) exceeds dimension size \\(10\\)\\.')
    if not is_narrow and (not is_ref) and (torch.device(device).type == 'cpu'):
        yield ErrorInput(SampleInput(make_arg((M,)), 0, 0, -1), error_type=RuntimeError, error_regex='start \\(0\\) \\+ length \\(-1\\) exceeds dimension size \\(10\\)\\.')
    else:
        yield ErrorInput(SampleInput(make_arg((M,)), 0, 0, -1), error_type=RuntimeError, error_regex='narrow\\(\\): length must be non-negative\\.')
    if is_narrow:
        yield ErrorInput(SampleInput(make_arg((L, M, S)), 1, make_arg(S, dtype=torch.int), 2), error_type=RuntimeError, error_regex='start must be an 0-dim integral Tensor\\.')
        yield ErrorInput(SampleInput(make_arg((L, M, S)), -3, make_arg((), dtype=torch.bool), 3), error_type=RuntimeError, error_regex='start must be an 0-dim integral Tensor\\.')

def sample_trapezoid(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    y_shape_x_shape_and_kwargs = [((2, 3), (2, 3), {}), ((2, 3), (2, 3), {'dim': 1}), ((6,), (6,), {}), ((6,), None, {}), ((2, 3), (1, 3), {}), ((3, 3), (3, 3), {}), ((3, 3), (3, 3), {'dim': -2}), ((5,), None, {'dx': 2.0}), ((2, 2), None, {'dx': 3.0})]
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    for (y_shape, x_shape, kwarg) in y_shape_x_shape_and_kwargs:
        y_tensor = make_arg(y_shape)
        if x_shape is not None:
            x_tensor = make_arg(x_shape)
            yield SampleInput(y_tensor, x_tensor, **kwarg)
        else:
            yield SampleInput(y_tensor, **kwarg)

def sample_cumulative_trapezoid(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    y_shape_x_shape_and_kwargs = [((2, 3), (2, 3), {}), ((2, 3), (2, 3), {'dim': 1}), ((6,), (6,), {}), ((6,), None, {}), ((2, 3), (1, 3), {}), ((3, 3), (3, 3), {}), ((3, 3), (3, 3), {'dim': -2}), ((5,), None, {'dx': 2.0}), ((2, 2), None, {'dx': 3.0})]
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    for (y_shape, x_shape, kwarg) in y_shape_x_shape_and_kwargs:
        y_tensor = make_arg(y_shape)
        if x_shape is not None:
            x_tensor = make_arg(x_shape)
            yield SampleInput(y_tensor, x_tensor, **kwarg)
        else:
            yield SampleInput(y_tensor, **kwarg)

def sample_unsqueeze(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    shapes_and_axes = [((3, 4, 5), 0), ((3, 4, 5), 1), ((3, 4, 5), 3), ((3, 4, 5), -1), ((3, 4, 5), -3), ((), 0), ((), -1), ((1,), 0), ((1,), -1)]
    for (shape, axis) in shapes_and_axes:
        tensor = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(tensor, axis)

def sample_inputs_nn_unfold(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    shapes = ((0, 1, 5, 5), (2, 3, 5, 5))
    kernel_sizes = (2, (2, 2), (2, 3))
    dilations = (1, 2, (1, 2))
    paddings = (0, 1, (1, 2))
    strides = (1, 2, (1, 2))
    cases = product(shapes, kernel_sizes, dilations, paddings, strides)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for (shape, kernel_size, dilation, padding, stride) in cases:
        tensor = make_arg(shape)
        yield SampleInput(tensor, kernel_size, dilation, padding, stride)
    yield SampleInput(make_arg((1, 1, 5, 5)), (3, 3))

def sample_inputs_squeeze(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    shapes_and_args = (((S, 1, S, 1), ()), ((1, 1, 1, 1), ()), ((1, 1, 1, 1), (0,)), ((S, 1, S, 1), (1,)), ((S, 1, S, 1), (-1,)), ((S, 1, S, 1), (2,)), ((S, 1, S, 1), (-2,)), ((), (0,)))
    for (shape, args) in shapes_and_args:
        tensor = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(tensor, args=args)

def sample_inputs_squeeze_multiple(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    shapes_and_args = (((1, 1, 1, 1), ()), ((S, 1, S, 1), (1,)), ((S, 1, S, 1), (-1,)), ((S, 1, S, 1), (1, 3)), ((S, 1, S, 1), (1, 2)), ((), (0,)))
    for (shape, dims) in shapes_and_args:
        tensor = make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
        yield SampleInput(tensor, dims)

def _squeeze_ref(x, axis=None):
    if False:
        return 10
    if x.ndim == 0:
        return x
    if isinstance(axis, Sequence):
        axis = tuple((a for a in axis if x.shape[a] == 1))
    if isinstance(axis, int) and x.shape[axis] != 1:
        return x
    return np.squeeze(x, axis)

def sample_inputs_nn_pad(op_info, device, dtype, requires_grad, mode, **kwargs):
    if False:
        while True:
            i = 10
    assert mode in ('constant', 'reflect', 'replicate', 'circular')
    if mode in ['reflect', 'replicate']:
        cases: tuple = (((1, 3), (1, 2)), ((1, 3), (0, 1)), ((0, 3, 3), (1, 2)), ((0, 3, 3), (0, 1)), ((1, 3, 3), (1, 2)), ((1, 3, 3), (0, 1)), ((1, 3, 3), (0, 2, 0, 1)), ((0, 3, 3, 3), (0, 2, 0, 1)), ((3, 3, 5, 5), (0, 2, 0, 1)), ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)), ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 4, 4), (-1, 1, -2, 1)))
    elif mode == 'constant':
        cases = (((1, 3), (1, 2)), ((1, 3), (0, 1)), ((1, 3), (0, 2, 0, 1)), ((0, 3, 3), (1, 2)), ((0, 3, 3), (0, 1)), ((0, 3, 3), (0, 2, 0, 1)), ((0, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 3), (1, 2)), ((1, 3, 3), (0, 1)), ((1, 3, 3), (0, 2, 0, 1)), ((1, 3, 3), (1, 1, 1, 1, 1, 1)), ((0, 3, 3, 3), (1, 2)), ((0, 3, 3, 3), (0, 1)), ((0, 3, 3, 3), (0, 2, 0, 1)), ((0, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((3, 3, 5, 5), (1, 2)), ((3, 3, 5, 5), (0, 1)), ((3, 3, 5, 5), (0, 2, 0, 1)), ((3, 3, 5, 5), (1, 1, 1, 1, 1, 1)), ((1, 3, 3, 3, 3), (1, 2)), ((1, 3, 3, 3, 3), (0, 1)), ((1, 3, 3, 3, 3), (0, 2, 0, 1)), ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 4, 4), (-1, 1, -2, 1)))
    elif dtype == torch.bool:
        cases = (((2, 3, 3), (1, 2)), ((1, 3, 3), (1, 2)))
    else:
        cases = (((0, 3, 3), (1, 2)), ((0, 3, 3), (0, 1)), ((1, 3, 3), (1, 2)), ((1, 3, 3), (0, 1)), ((0, 3, 3, 3), (0, 2, 0, 1)), ((3, 3, 5, 5), (0, 2, 0, 1)), ((1, 3, 3, 3, 3), (1, 1, 1, 1, 1, 1)), ((1, 3, 4, 4), (-1, 1, -2, 1)))
    make_inp = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    if mode == 'constant':
        yield SampleInput(make_inp((1, 3, 3)), args=((2, 2),))
    if mode in ['reflect', 'replicate', 'circular']:
        for (shape, pad) in cases:
            yield SampleInput(make_inp(shape), args=(pad, mode))
    else:
        for pad_value in (1.0, 2.0):
            for (shape, pad) in cases:
                yield SampleInput(make_inp(shape), args=(pad, mode, pad_value))

def sample_inputs_constant_pad_nd(op_info, device, dtype, *args, **kwargs):
    if False:
        return 10
    nn_samples = sample_inputs_nn_pad(op_info, device, dtype, *args, mode='constant', **kwargs)
    from torch._prims_common import dtype_to_type
    scalar_type = dtype_to_type(dtype)

    def drop_mode_argument(input, pad, mode=None, value=None):
        if False:
            for i in range(10):
                print('nop')
        if value is None:
            return SampleInput(input, args=(pad,))
        else:
            return SampleInput(input, args=(pad, scalar_type(value)))
    for sample in nn_samples:
        yield drop_mode_argument(sample.input, *sample.args, **sample.kwargs)

def sample_inputs_repeat_interleave(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_input(()), repeats=2)
    yield SampleInput(make_input((2, 3, 4)), repeats=2)
    yield SampleInput(make_input((2, 3, 4)), repeats=2, dim=1)
    yield SampleInput(make_input((2, 3, 4)), repeats=torch.arange(3, device=device), dim=1)

def sample_inputs_stft(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15

    def mt(shape, **kwargs):
        if False:
            while True:
                i = 10
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad, **kwargs)
    yield SampleInput(mt(100), n_fft=10, return_complex=True)
    yield SampleInput(mt(100), n_fft=10, return_complex=False)
    if dtype.is_complex:
        yield SampleInput(mt(100), n_fft=10)
    for center in [False, True]:
        yield SampleInput(mt(10), n_fft=7, center=center, return_complex=True)
        yield SampleInput(mt((10, 100)), n_fft=16, hop_length=4, center=center, return_complex=True)
    window = mt(16, low=0.5, high=2.0)
    yield SampleInput(mt((2, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True, center=center))
    yield SampleInput(mt((3, 100)), kwargs=dict(n_fft=16, window=window, return_complex=True, center=center))
    if not dtype.is_complex:
        yield SampleInput(mt((10, 100)), n_fft=16, window=window, onesided=False, return_complex=True)

def sample_inputs_istft(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def mt(shape, **kwargs):
        if False:
            print('Hello World!')
        real_shape = shape if dtype.is_complex else shape + (2,)
        return make_arg(real_shape, **kwargs)
    yield SampleInput(mt((10, 2)), kwargs=dict(n_fft=10))
    yield SampleInput(mt((6, 3)), kwargs=dict(n_fft=6, onesided=False))
    yield SampleInput(mt((6, 4)), kwargs=dict(n_fft=10, onesided=True))
    for center in [False, True]:
        yield SampleInput(mt((10, 10, 6)), kwargs=dict(n_fft=10, center=center))
        yield SampleInput(mt((1, 9, 10)), kwargs=dict(n_fft=16, hop_length=4, center=center))
    window = make_arg(10, low=0.5, high=2.0)
    yield SampleInput(mt((10, 10, 6)), kwargs=dict(n_fft=10, window=window, center=center, return_complex=dtype.is_complex))
    yield SampleInput(mt((10, 10, 10)), kwargs=dict(n_fft=10, window=window[:8], win_length=8, center=center, return_complex=True))
    real_window = window if not dtype.is_complex else window.real
    yield SampleInput(mt((10, 5, 6)), kwargs=dict(n_fft=8, window=real_window[:8], center=center))

def sample_inputs_ormqr(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_input = partial(make_tensor, dtype=dtype, device=device, low=-1, high=1)
    batches = [(), (0,), (2,), (2, 1)]
    ns = [5, 2, 0]
    tf = [True, False]
    for (batch, (m, n), left, transpose) in product(batches, product(ns, ns), tf, tf):
        input = make_input((*batch, m, n))
        (reflectors, tau) = torch.geqrf(input)
        reflectors.requires_grad_(requires_grad)
        tau.requires_grad_(requires_grad)
        other_matrix_shape = (m, n) if left else (n, m)
        other = make_input((*batch, *other_matrix_shape), requires_grad=requires_grad)
        yield SampleInput(reflectors, tau, other, left=left, transpose=transpose)

def sample_inputs_cholesky_solve(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    cholesky_inverse_samples = sample_inputs_linalg_cholesky_inverse(op_info, device, dtype, requires_grad=False)
    for sample in cholesky_inverse_samples:
        psd_matrix = sample.input
        sample.input = make_tensor(psd_matrix.shape, dtype=dtype, device=device, requires_grad=requires_grad, low=None, high=None)
        sample.args = (psd_matrix.requires_grad_(requires_grad),)
        yield sample

def sample_inputs_lu(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_fullrank_matrices_with_distinct_singular_values, dtype=dtype, device=device, requires_grad=requires_grad)
    batch_shapes = ((), (3,), (3, 3))
    for (batch_shape, get_infos, size_delta) in product(batch_shapes, (True, False), (-2, -1, 0, +1, +2)):
        shape = batch_shape + (S + size_delta, S)
        input = make_arg(*shape)
        yield SampleInput(input, args=(True, get_infos))

def sample_inputs_lu_unpack(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    def out_fn(output):
        if False:
            i = 10
            return i + 15
        return (output[1], output[2])
    for lu_sample in sample_inputs_linalg_lu(op_info, device, dtype, requires_grad, **kwargs):
        (lu_data, pivots) = torch.linalg.lu_factor(lu_sample.input)
        lu_data.requires_grad_(requires_grad)
        yield SampleInput(lu_data, pivots).with_metadata(output_process_fn_grad=out_fn)

def sample_inputs_roll(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    args = ((0, 0), (1, 2), (0, 2), (2, 0), (-1, 0), (10000, 1), (2,), ((1, 2, -1), (0, 1, 2)))
    for arg in args:
        yield SampleInput(make_arg((0, 0, 0)), args=arg)
        yield SampleInput(make_arg((S, S, S)), args=arg)
    yield SampleInput(make_arg(()), args=(10,))

def error_inputs_roll(op_info, device, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    err_msg1 = '`shifts` required'
    s1 = SampleInput(make_arg((S,)), ())
    yield ErrorInput(s1, error_regex=err_msg1)
    err_msg2 = 'shifts and dimensions must align'
    s2 = SampleInput(make_arg((S, S)), (2, 1), 0)
    yield ErrorInput(s2, error_regex=err_msg2)
    err_msg3 = 'out of range'
    s3 = SampleInput(make_arg((S,)), 0, 2)
    yield ErrorInput(s3, error_regex=err_msg3, error_type=IndexError)
    err_msg4 = 'Dimension specified as 0'
    s4 = SampleInput(make_arg(()), 0, 0)
    yield ErrorInput(s4, error_regex=err_msg4, error_type=IndexError)

def sample_inputs_rot90(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    args = itertools.product(range(-5, 6), [(0, 1), (1, 2), (1, -1)])
    yield SampleInput(make_arg((S, S, S)))
    for arg in args:
        yield SampleInput(make_arg((S, S, S)), args=arg)

def error_inputs_rot90(op_info, device, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    err_msg1 = 'expected total rotation dims'
    s1 = SampleInput(make_arg((S, S)), dims=(0,))
    yield ErrorInput(s1, error_regex=err_msg1)
    err_msg2 = 'expected total dims >= 2'
    s2 = SampleInput(make_arg((S,)))
    yield ErrorInput(s2, error_regex=err_msg2)
    err_msg3 = 'expected rotation dims to be different'
    s3 = SampleInput(make_arg((S, S)), dims=(1, 1))
    yield ErrorInput(s3, error_regex=err_msg3)

def sample_inputs_std_var(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    tensor_nd = partial(make_tensor, (S, S, S), device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_1d = partial(make_tensor, (S,), device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(tensor_nd())
    yield SampleInput(tensor_nd(), dim=1)
    yield SampleInput(tensor_nd(), dim=1, unbiased=True, keepdim=True)
    yield SampleInput(tensor_1d(), dim=0, unbiased=True, keepdim=True)
    yield SampleInput(tensor_1d(), dim=0, unbiased=False, keepdim=False)
    yield SampleInput(tensor_nd(), dim=(1,), correction=1.3)
    yield SampleInput(tensor_nd(), dim=(1,), correction=S // 2)
    yield SampleInput(tensor_nd(), dim=None, correction=0, keepdim=True)
    yield SampleInput(tensor_nd(), dim=None, correction=None)
    yield SampleInput(tensor_nd(), correction=0, keepdim=True)
    yield SampleInput(make_tensor(3, 4, 5, device=device, dtype=dtype, requires_grad=requires_grad), dim=-3)

def sample_inputs_std_var_unbiased(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)), True)
    yield SampleInput(make_arg((S,)), False)

def _generate_correlation_inputs(device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    shapes = [(2,), (1, 2), (3, 2), (2, 3)]
    for shape in shapes:
        yield make_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad)

def sample_inputs_corrcoef(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    return (SampleInput(t) for t in _generate_correlation_inputs(device, dtype, requires_grad))

def sample_inputs_cov(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    for t in _generate_correlation_inputs(device, dtype, requires_grad):
        yield SampleInput(t)
        num_observations = t.numel() if t.ndimension() < 2 else t.size(1)
        fweights = make_tensor((num_observations,), dtype=torch.int, device=device, low=1, high=10)
        aweights = make_tensor((num_observations,), dtype=torch.float, device=device, low=0, high=1, requires_grad=requires_grad)
        for (correction, fw, aw) in product(range(num_observations), [None, fweights], [None, aweights]):
            yield SampleInput(t.clone().requires_grad_(requires_grad), correction=correction, fweights=fw, aweights=aw)

def error_inputs_cov(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    a = torch.rand(S, device=device)
    yield ErrorInput(SampleInput(torch.rand(S, S, S, device=device)), error_regex='expected input to have two or fewer dimensions')
    yield ErrorInput(SampleInput(a, fweights=torch.rand(S, S, device=device)), error_regex='expected fweights to have one or fewer dimensions')
    yield ErrorInput(SampleInput(a, aweights=torch.rand(S, S, device=device)), error_regex='expected aweights to have one or fewer dimensions')
    yield ErrorInput(SampleInput(a, fweights=torch.rand(S, device=device)), error_regex='expected fweights to have integral dtype')
    yield ErrorInput(SampleInput(a, aweights=torch.tensor([1, 1], device=device)), error_regex='expected aweights to have floating point dtype')
    yield ErrorInput(SampleInput(a, fweights=torch.tensor([1], device=device)), error_regex='expected fweights to have the same numel')
    yield ErrorInput(SampleInput(a, aweights=torch.rand(1, device=device)), error_regex='expected aweights to have the same numel')
    yield ErrorInput(SampleInput(a, fweights=torch.tensor([-1, -2, -3, -4, -5], device=device)), error_regex='fweights cannot be negative')
    yield ErrorInput(SampleInput(a, aweights=torch.tensor([-1.0, -2.0, -3.0, -4.0, -5.0], device=device)), error_regex='aweights cannot be negative')

def sample_inputs_permute(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = [((1, 2, 3, 4), (0, 2, 3, 1)), ((1, 2, 3, 4), (0, -2, -1, 1)), ((), ()), ((1, 2, 3, 4), (2, 1, 3, 0))]
    for (shape, args) in cases:
        yield SampleInput(make_arg(shape), args=(args,))

def reference_inputs_permute(op, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    yield from sample_inputs_permute(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((), ()), ((1,), (0,)), ((2, 2), (1, 0)), ((2, 2), (0, 1)), ((2, 0, 1), (0, 2, 1)), ((3, 4, 2), (2, 1, 0)), ((3, 4, 2), (1, 0, 2)), ((3, 4, 2), (0, 1, 2)))
    for (shape, permutation) in cases:
        for p in itertools.permutations(permutation):
            a = make_arg(shape).permute(p)
            yield SampleInput(a, args=(permutation,))
            a = make_arg(shape, noncontiguous=True).permute(p)
            yield SampleInput(a, args=(permutation,))

def error_inputs_softshrink(op, device, **kwargs):
    if False:
        while True:
            i = 10
    yield ErrorInput(SampleInput(make_tensor((1,), dtype=torch.float, device=device), kwargs={'lambd': -0.5}), error_regex='lambda must be greater or equal to 0, but found to be -0.5')

def sample_inputs_softshrink(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for lbda in (0.0, 0.5):
        yield SampleInput(make_arg(S, S), kwargs={'lambd': lbda})
    yield from sample_inputs_elementwise_unary(op_info, device, dtype, requires_grad)

def sample_inputs_hardshrink(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for lbda in (-0.5, 0.0, 0.5):
        yield SampleInput(make_arg(S, S), kwargs={'lambd': lbda})
    yield from sample_inputs_elementwise_unary(op_info, device, dtype, requires_grad)

def sample_inputs_hardtanh(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for (max_val, min_val) in ((-0.5, 0.5), (0.5, -0.5), (0.0, 0.0)):
        yield SampleInput(make_arg(S, S), kwargs={'min_val': min_val, 'max_val': max_val})
    yield from sample_inputs_elementwise_unary(op_info, device, dtype, requires_grad)

def sample_inputs_einsum(op_info, device, dtype, requires_grad=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    def c(t):
        if False:
            print('Hello World!')
        return t.clone().requires_grad_(requires_grad)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    x = make_arg((3,))
    y = make_arg((4,))
    A = make_arg((2, 3))
    B = make_arg((1, 3))
    C = make_arg((1, 2, 3))
    D = make_arg((1, 3, 4))
    E = make_arg((4, 4))
    H = make_arg((3, 3))
    I = make_arg((1, 3, 1))
    yield SampleInput([c(x)], 'i->')
    yield SampleInput([c(x), c(y)], 'i,j->ij')
    yield SampleInput([c(A)], 'ij->i')
    yield SampleInput([c(A), c(B)], 'ij,kj->ik')
    yield SampleInput([c(A), c(E)], 'ij,Ab->ijAb')
    yield SampleInput([c(C), c(D)], 'aij,ajk->aik')
    yield SampleInput([c(D), c(E)], 'aij,jk->aik')
    yield SampleInput([c(C), c(B)], 'ijk,ik->j')
    yield SampleInput([c(I)], 'iji->j')
    yield SampleInput([c(H)], 'i...->...')
    yield SampleInput([c(C), c(x)], '...ik, ...j -> ij')

def sample_inputs_flip(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((S, M, S), (S, 0, M))
    all_dims = ((0, 1, 2), (0,), (0, 2), (-1,), ())
    for (size, dims) in product(sizes, all_dims):
        yield SampleInput(make_arg(size), kwargs={'dims': dims})

def sample_inputs_fliplr_flipud(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    shapes = [(S, M, S), (S, 0, M)]
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    return (SampleInput(make_arg(shape, low=None, high=None)) for shape in shapes)

def error_inputs_fliplr(op, device, **kwargs):
    if False:
        while True:
            i = 10
    yield ErrorInput(SampleInput(make_tensor((1,), dtype=torch.float, device=device)), error_regex='Input must be >= 2-d.')

def error_inputs_flipud(op, device, **kwargs):
    if False:
        return 10
    yield ErrorInput(SampleInput(make_tensor((), dtype=torch.float, device=device)), error_regex='Input must be >= 1-d.')

def sample_inputs_clamp(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    shape = (S, M, S)
    yield SampleInput(make_arg(shape), args=(make_arg(shape), make_arg(shape)))
    yield SampleInput(make_arg(shape), args=(make_arg(shape[1:]), make_arg(shape[1:])))
    yield SampleInput(make_arg(shape), args=(make_arg((S, 1, S)),))
    yield SampleInput(make_arg(shape), args=(None, make_arg(shape)))
    yield SampleInput(make_arg(shape), args=(make_arg(shape), None))

def reference_inputs_elementwise_ternary(op, device, dtype, requires_grad, *, sample_inputs_func, supports_scalars=False, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_func(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_scalar_tensor = partial(make_tensor, (), device='cpu', dtype=dtype, requires_grad=requires_grad)
    supported_dtypes = op.supported_dtypes(device)
    cases = (((4, 4), (4, 4), (4, 4)), ((4, 4), (1, 4, 4), (4, 4)), ((4, 4), (1, 4, 4), (4, 1, 4)), ((4, 4, 1), (1, 4, 4), (4, 4)), ((4, 1), (1, 4, 4), (1, 4)), ((4, 4), (), (4, 4)), ((4, 4), (), ()), ((), (4, 4), (1, 4, 4)))
    for (a, b, c) in cases:
        yield SampleInput(make_arg(a), args=(make_arg(b), make_arg(c)))
        yield SampleInput(make_arg(a, noncontiguous=True), args=(make_arg(b).transpose(0, -1), make_arg(c, noncontiguous=True).transpose(0, -1)))
    if supports_scalars:
        cases = [((), 1, 2), ((), 1.0, 2), ((4, 4), 1.0, 2), ((3, 4), make_scalar_tensor(), make_scalar_tensor())]
        if torch.complex64 in supported_dtypes:
            cases.extend([((3, 1, 4), complex(1, 2), 3.0)])
        for (a, b, c) in cases:
            yield SampleInput(make_arg(a), args=(b, c))
    if torch.float in supported_dtypes and torch.long in supported_dtypes:
        a = make_arg((), dtype=torch.long)
        b = make_arg((1, 4), dtype=torch.float)
        c = make_arg((3, 4))
        cases = ((a, b, c), (c, a, b))
        for (a, b, c) in cases:
            yield SampleInput(a, args=(b, c))
    if dtype.is_floating_point or dtype.is_complex:
        nan = float('nan') if dtype.is_floating_point else complex(float('nan'), float('nan'))
        a = make_arg((12,))
        a[4] = nan
        a[7] = nan
        b = make_arg((12,))
        b[1] = nan
        b[7] = nan
        c = make_arg((12,))
        c[9] = nan
        yield SampleInput(a, args=(b, c))

def _clamp_min_numpy(a, min=None):
    if False:
        print('Hello World!')
    return np.maximum(a, min)

def _clamp_max_numpy(a, max=None):
    if False:
        print('Hello World!')
    return np.minimum(a, max)

def _clamp_numpy(a, min=None, max=None):
    if False:
        print('Hello World!')
    if min is None:
        return np.minimum(a, max)
    if max is None:
        return np.maximum(a, min)
    return np.minimum(max, np.maximum(a, min))

def sample_inputs_cumprod(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')

    def make_arg(shape):
        if False:
            return 10
        return make_tensor(shape, dtype=dtype, device=device, low=-1, high=+1, requires_grad=requires_grad)

    def prod_zeros(dim_select):
        if False:
            print('Hello World!')
        assert len(dim_select) == 2
        result = make_arg(3 * (S,))
        result.narrow(dim_select[0], 0, 1).narrow(dim_select[1], 1, 1).zero_()
        result.narrow(dim_select[0], 2, 1).narrow(dim_select[1], 3, 1).zero_()
        result.narrow(dim_select[0], 4, 1).narrow(dim_select[1], 3, 1).zero_()
        return result
    for dim in range(3):
        yield SampleInput(make_arg((S, S, S)), args=(dim,))
    for size in [(), (1,), (0,)]:
        yield SampleInput(make_arg(size), args=(0,))
    yield SampleInput(prod_zeros([0, 1]), args=(1,))
    yield SampleInput(prod_zeros([0, 2]), args=(1,))
    yield SampleInput(prod_zeros([1, 2]), args=(1,))
    yield SampleInput(prod_zeros([1, 2]), args=(1,), kwargs={'dtype': dtype})

def sample_inputs_view_as_complex(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield SampleInput(make_tensor((S, 2), dtype=dtype, device=device, requires_grad=requires_grad))

def sample_inputs_view_as_real(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    sizes = ((S, S), ())
    return (SampleInput(make_arg(size)) for size in sizes)

def error_inputs_complex(op_info, device, is_ref=False, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=torch.float32, device=device)
    if is_ref:
        error_float = 'Expected both inputs to be Half, Float or Double tensors but got torch.float32 and torch.int32'
        error_dtype = 'Expected object of scalar type torch.float32 but got scalar type torch.float64 for second argument'
        error_out = 'Expected out tensor to have dtype torch.complex128 but got torch.complex64 instead'
    else:
        error_float = 'Expected both inputs to be Half, Float or Double tensors but got Float and Int'
        error_dtype = 'Expected object of scalar type Float but got scalar type Double for second argument'
        error_out = "Expected object of scalar type ComplexDouble but got scalar type ComplexFloat for argument 'out'"
    yield ErrorInput(SampleInput(make_arg(M, S), make_arg(M, S, dtype=torch.int)), error_type=RuntimeError, error_regex=error_float)
    yield ErrorInput(SampleInput(make_arg(M, S), make_arg(M, S, dtype=torch.float64)), error_type=RuntimeError, error_regex=error_dtype)
    yield ErrorInput(SampleInput(make_arg(M, S, dtype=torch.float64), make_arg(M, S, dtype=torch.float64), out=make_arg(M, S, dtype=torch.complex64)), error_type=RuntimeError, error_regex=error_out)

def sample_inputs_logaddexp(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shape = (S, S)
    yield SampleInput(make_arg(shape), make_arg(shape))

def sample_inputs_prod(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')

    def make_arg(shape):
        if False:
            while True:
                i = 10
        return make_tensor(shape, dtype=dtype, device=device, low=-1, high=+1, requires_grad=requires_grad)

    def prod_single_zero():
        if False:
            return 10
        result = make_arg(2 * (S,))
        result[0, 1] = 0
        return result
    for sample in sample_inputs_cumprod(op_info, device, dtype, requires_grad):
        yield SampleInput(sample.input.clone().requires_grad_(requires_grad))
        yield sample
    for sample in sample_inputs_cumprod(op_info, device, dtype, requires_grad):
        sample.kwargs['keepdim'] = True
        yield sample
    yield SampleInput(prod_single_zero())
    yield SampleInput(make_arg((3, 3, 3)), args=(1,))
    yield SampleInput(make_arg((3, 3, 3)), args=(1,), kwargs={'keepdim': True})
    yield SampleInput(make_arg((3, 0)), args=(1,))
    yield SampleInput(make_arg((3, 0)), args=(1,), kwargs={'keepdim': True})
    zero = make_arg(())
    zero.zero_()
    yield SampleInput(zero.clone().requires_grad_(requires_grad))
    yield SampleInput(zero.clone().requires_grad_(requires_grad), args=(0,))
    yield SampleInput(zero.clone().requires_grad_(requires_grad), args=(0,), kwargs={'keepdim': True})

def error_inputs_neg(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    si = SampleInput(torch.tensor((False, True), device=device))
    msg = 'Negation, the `\\-` operator, on a bool tensor is not supported. If you are trying to invert a mask, use the `\\~` or `logical_not\\(\\)` operator instead.'
    yield ErrorInput(si, error_regex=msg)

def sample_inputs_diag(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    yield SampleInput(make_arg(M))
    tensors = (make_arg((M, M)), make_arg((3, 5)), make_arg((5, 3)))
    args = ((), (2,), (-2,), (1,), (2,))
    for (tensor, arg) in product(tensors, args):
        yield SampleInput(tensor.clone().requires_grad_(requires_grad), *arg)

def reference_inputs_diagonal_diag_embed(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    yield from sample_inputs_diagonal_diag_embed(op_info, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes1d = ((0,), (1,))
    shapes2d = ((L, M),)
    shapes3d = ((L, M, S),)
    kwargs1d = {}
    kwargs2d = (dict(dim1=1, dim2=0), dict(dim1=-2, dim2=-1), dict(offset=100))
    kwargs3d = kwargs2d + (dict(offset=-1, dim1=0, dim2=2),)
    samples1d = product(shapes1d, kwargs1d)
    samples2d = product(shapes2d, kwargs2d)
    samples3d = product(shapes3d, kwargs3d)
    for (shape, kwargs) in chain(samples1d, samples2d, samples3d):
        if 'diagonal' in op_info.name:
            if shape in ((0,), (1,)):
                continue
        yield SampleInput(input=make_arg(shape), kwargs=kwargs)

def sample_inputs_diagonal_scatter(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes_2d = ((M, M), (3, 5), (5, 3))
    shapes_3d = ((M, M, M),)
    args_2d = ((), (2,), (-2,), (1,))
    args_3d = ((1, 1, 2), (2, 0, 1), (-2, 0, 1))
    for (input_shape, arg) in chain(product(shapes_2d, args_2d), product(shapes_3d, args_3d)):
        input_ = make_arg(input_shape)
        if not isinstance(arg, tuple):
            arg_tuple = (arg,)
        else:
            arg_tuple = arg
        src_shape = input_.diagonal(*arg_tuple).size()
        src = make_arg(src_shape)
        yield SampleInput(input_, args=(src, *arg_tuple))

def sample_inputs_to_sparse(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S))).with_metadata(output_process_fn_grad=lambda x: x.to_dense())
    yield SampleInput(make_arg((S, S)), 1).with_metadata(output_process_fn_grad=lambda x: x.to_dense())

def sample_inputs_cross_entropy(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    (batch_size, num_classes) = shape = (2, 3)
    reductions = ('mean', 'sum', 'none')
    input_shape_and_kwargs: List[Tuple[Tuple[int, ...], Dict[str, Any]]] = [(shape, {}), ((*shape, 1), {}), ((*shape, 1, 2), {}), ((*shape, 1, 2, 3), {}), *[(shape, dict(reduction=reduction)) for reduction in reductions], *[(shape, dict(weight=make_tensor((num_classes,), device=device, dtype=dtype), reduction=reduction)) for reduction in reductions], (shape, dict(ignore_index=1))]
    for ((input_shape, kwargs), probabilities_target) in itertools.product(input_shape_and_kwargs, (False, True)):
        input = make_tensor(input_shape, device=device, dtype=dtype, requires_grad=requires_grad)
        if probabilities_target:
            if 'ignore_index' in kwargs:
                continue
            target = make_tensor(input_shape, low=0, high=1, device=device, dtype=dtype, requires_grad=requires_grad)
        else:
            target = make_tensor((batch_size, *input_shape[2:]), low=0, high=num_classes, device=device, dtype=torch.long)
            if 'ignore_index' in kwargs and torch.all(target == kwargs['ignore_index']):
                target[0] = random.sample(sorted(set(range(num_classes)) - {kwargs['ignore_index']}), 1)[0]
        yield SampleInput(input, target, **kwargs)

def sample_inputs_logit(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    (low, high) = op_info.domain
    if dtype.is_floating_point or dtype.is_complex:
        domain_eps = op_info._domain_eps if dtype != torch.float16 else 0.03
        low = low + domain_eps
        high = high - domain_eps
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S, S)))
    yield SampleInput(make_arg((S, S, S)), 0.2)
    yield SampleInput(make_arg(()))
    yield SampleInput(make_arg(()), 0.2)

def sample_inputs_isin(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((L,)), args=(make_arg((S,)),))
    yield SampleInput(make_arg((S,)), args=(make_arg((L,)),))

def sample_inputs_masked_scatter(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))))
    yield SampleInput(make_arg((S, S)), args=(torch.randn((S,), device=device) > 0, make_arg((S, S))))
    yield SampleInput(make_arg((S, S)), args=(bernoulli_scalar().to(device), make_arg((S, S))))
    yield SampleInput(make_arg((S,)), args=(torch.randn(S, S, device=device) > 0, make_arg((S, S))), broadcasts_input=True)

def error_inputs_masked_scatter(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float)
    for mask_dtype in [torch.float, torch.uint8]:
        yield ErrorInput(SampleInput(make_arg(1, 3), args=(torch.ones(1, 3, device=device, dtype=mask_dtype), make_arg(3, 4))), error_regex='masked_scatter_ only supports boolean masks')

def sample_inputs_masked_fill(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, 10))
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, make_arg(())))
    yield SampleInput(make_arg((S, S)), args=(torch.randn(S, device=device) > 0, 10))
    yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, 10))
    yield SampleInput(make_arg(()), args=(torch.randn((), device=device) > 0, make_arg(())))
    yield SampleInput(make_arg((S, S)), args=(torch.randn((), device=device) > 0, 10))
    yield SampleInput(make_arg((S,)), args=(torch.randn(S, S, device=device) > 0, make_arg(())), broadcasts_input=True)
    yield SampleInput(make_arg((S,)), args=(torch.randn(S, S, device=device) > 0, 10), broadcasts_input=True)
    if torch.device(device).type == 'cuda':
        yield SampleInput(make_arg((S, S)), args=(torch.randn(S, S, device=device) > 0, torch.randn(())))

def error_inputs_masked_fill(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=torch.float, requires_grad=False)
    yield ErrorInput(SampleInput(make_arg((2, 2)), args=(make_arg(()) > 0, make_arg((1,)))), error_regex='only supports a 0-dimensional value tensor, but got tensor with 1 dimension')
    yield ErrorInput(SampleInput(make_arg((2, 2)), args=(make_arg(()) > 0, 1j)), error_regex='value cannot be converted to type .* without overflow')
    yield ErrorInput(SampleInput(torch.ones(2, dtype=torch.long, device=device), args=(make_arg(()) > 0, torch.tensor(1j, device=device))), error_regex='value cannot be converted to type .* without overflow')
    if torch.device(device).type == 'cuda':
        yield ErrorInput(SampleInput(torch.randn((S, S), device='cpu'), args=(torch.randn(S, S, device='cpu') > 0, torch.randn((), device='cuda'))), error_regex='to be on same device')

def sample_inputs_masked_select(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=None, high=None)
    yield SampleInput(make_arg((M, M)), torch.randn(M, M, device=device) > 0)
    yield SampleInput(make_arg((M, M)), torch.randn((M,), device=device) > 0)
    yield SampleInput(make_arg((M,)), torch.randn((M, M), device=device) > 0)
    yield SampleInput(make_arg((M, 1, M)), torch.randn((M, M), device=device) > 0)
    yield SampleInput(make_arg(()), torch.tensor(1, device=device, dtype=torch.bool))
    yield SampleInput(make_arg((M, M)), torch.tensor(1, device=device, dtype=torch.bool))
    yield SampleInput(make_arg(()), torch.randn((M, M), device=device) > 0)

def sample_inputs_matrix_exp(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)))
    yield SampleInput(make_arg((S, S, S)))

def sample_inputs_matmul(op_info, device, dtype, requires_grad, is_rmatmul=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    test_cases = (((L,), (L,)), ((S, M), (M,)), ((M,), (M, S)), ((S, M), (M, S)), ((S, 0), (0, M)), ((S, S, M), (M,)), ((S, S, M), (M, S)), ((S, S, 0), (0, S)), ((M,), (S, M, S)), ((S, M), (S, M, S)), ((0, 0), (S, 0, 0)), ((S, S, M, M), (S, S, M, S)), ((S, S, M, M), (M,)), ((M,), (S, S, M, S)), ((S, S, S), (1, S, S)))
    for (lhs_shape, rhs_shape) in test_cases:
        lhs = make_arg(lhs_shape)
        rhs = make_arg(rhs_shape)
        if not is_rmatmul:
            yield SampleInput(lhs, rhs)
        else:
            yield SampleInput(rhs, lhs)

def sample_inputs_meshgrid(op_info: OpInfo, device: torch.device, dtype: torch.dtype, requires_grad: bool, *, variant: str, **kwargs) -> List[SampleInput]:
    if False:
        return 10
    if variant == 'variadic':

        def make_inputs(tensors: List[torch.Tensor]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, ...]]:
            if False:
                i = 10
                return i + 15
            return tensors
    elif variant == 'list':

        def make_inputs(tensors: List[torch.Tensor]) -> Tuple[Union[torch.Tensor, List[torch.Tensor]], Tuple[torch.Tensor, ...]]:
            if False:
                while True:
                    i = 10
            return [tensors]
    else:
        raise ValueError(f'Unsupported variant, must be one of {{"variadic", "list"}}. Got "{variant}".')
    SCALAR = torch.Size([])
    VECTOR = torch.Size([3])
    test_cases: List[List[torch.Size]] = [[SCALAR], [VECTOR], [VECTOR, SCALAR], [VECTOR, SCALAR, VECTOR], [VECTOR, SCALAR, VECTOR, SCALAR]]
    for (shapes, indexing) in itertools.product(test_cases, {'xy', 'ij'}):
        args = make_inputs([make_tensor(shape, dtype=dtype, device=device, requires_grad=requires_grad) for shape in shapes])
        yield SampleInput(*args, indexing=indexing)

def sample_inputs_mvlgamma(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    tensor_shapes = ((S, S), ())
    ns = (1, 2, 3, 4, 5)

    def compute_min_val(p):
        if False:
            print('Hello World!')
        return (p - 1.0) / 2
    for (shape, n) in product(tensor_shapes, ns):
        min_val = compute_min_val(n)
        if not dtype.is_floating_point:
            min_val += 1
        else:
            min_val += 2 * torch.finfo(dtype).eps
        yield SampleInput(make_arg(shape, low=min_val), args=(n,))

def skips_mvlgamma(skip_redundant=False):
    if False:
        for i in range(10):
            print('nop')
    skips = (DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_float_domains'), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_extremal'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.float16, torch.int8)), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=(torch.int8,)))
    if skip_redundant:
        skips = skips + (DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon'))
    return skips

def make_mvlgamma_opinfo(variant_test_name, domain, skips, sample_kwargs):
    if False:
        for i in range(10):
            print('nop')
    return UnaryUfuncInfo('mvlgamma', ref=reference_mvlgamma if TEST_SCIPY else None, aliases=('special.multigammaln',), variant_test_name=variant_test_name, domain=domain, decorators=(precisionOverride({torch.float16: 0.05}),), dtypes=all_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16), sample_inputs_func=sample_inputs_mvlgamma, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=skips, sample_kwargs=sample_kwargs)

def sample_inputs_cumulative_ops(op_info, device, dtype, requires_grad, supports_dtype_kwargs=True, **kwargs):
    if False:
        print('Hello World!')

    def _make_tensor_helper(shape, low=None, high=None):
        if False:
            return 10
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)
    yield SampleInput(_make_tensor_helper((S, S, S)), 0)
    yield SampleInput(_make_tensor_helper((S, S, S)), 1)
    yield SampleInput(_make_tensor_helper(()), 0)
    if supports_dtype_kwargs:
        yield SampleInput(_make_tensor_helper((S, S, S)), 1, dtype=dtype)

def sample_inputs_unfold(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    test_cases = (((), (0, 1, 1)), ((S, S, S, S), (0, 3, 1)), ((S, S, S, S), (1, 3, 1)), ((S, S, S, S), (2, 3, 1)), ((S, S, S, S), (3, 3, 1)), ((S, S, S, S), (0, 3, 2)), ((S, S, S, S), (1, 3, 2)), ((S, S, S, S), (2, 3, 2)), ((S, S, S, S), (3, 3, 2)), ((S, S, S, S), (0, 4, 1)), ((S, S, S, S), (1, 4, 1)), ((S, S, S, S), (2, 4, 1)), ((S, S, S, S), (3, 4, 1)), ((M,), (0, 3, 1)), ((M,), (0, 3, 2)), ((M,), (0, 3, 3)), ((1000,), (0, 3, 11)), ((1000,), (0, 2, 27)), ((10, 10), (0, 1, 2)), ((10, 10), (1, 2, 3)), ((10, 10), (1, 2, 2)), ((S, S, S), (2, 3, 2)))
    for (shape, arguments) in test_cases:
        yield SampleInput(make_tensor(shape, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad), *arguments)

def sample_inputs_split(op_info, device, dtype, requires_grad, *, list_args=False, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    if list_args:
        cases = (((S, S, S), (torch.Size([int(S / 3), S - int(S / 3) * 2, int(S / 3)]),)), ((S, S, S), (torch.Size([int(S / 2), S - int(S / 2) * 2, int(S / 2)]), 2)), ((S, S, S), (torch.Size([int(S / 2), S - int(S / 2) * 2, int(S / 2)]), -2)))
    else:
        cases = (((S, S, S), (2,)), ((S, S, S), (S, 1)))
    for (shape, args) in cases:
        yield SampleInput(make_arg(shape), args=args)

def sample_inputs_split_with_sizes(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    cases = (((S, S, S), (torch.Size([int(S / 3), S - int(S / 3) * 2, int(S / 3)]),)), ((S, S, S), (torch.Size([int(S / 3), S - int(S / 3), 0]),)), ((S, S, S), (torch.Size([int(S / 3), S - int(S / 3) * 2, int(S / 3)]), 2)), ((S, S, S), (torch.Size([int(S / 3), S - int(S / 3) * 2, int(S / 3)]), -2)))
    for (shape, args) in cases:
        yield SampleInput(make_arg(shape), args=args)

def sample_inputs_msort(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10

    def apply_grad(t):
        if False:
            while True:
                i = 10
        if dtype in floating_types_and(torch.float16, torch.bfloat16):
            t.requires_grad_(requires_grad)

    def large_1d_unique(dtype, device):
        if False:
            for i in range(10):
                print('nop')
        res = torch.randperm(L * L * L, dtype=torch.int64, device=device)
        res = res.to(dtype)
        apply_grad(res)
        return res
    yield SampleInput(large_1d_unique(dtype, device))
    yield SampleInput(make_tensor((S, M, S), dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad))

def sample_inputs_lerp(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S)), make_arg((S, S)), 0.4)
    yield SampleInput(make_arg((S, S)), make_arg((S,)), 0.4)
    yield SampleInput(make_arg(()), make_arg(()), 0.4)
    yield SampleInput(make_arg((S, S)), make_arg(()), 0.4)
    yield SampleInput(make_arg((S, S)), make_arg((S,)), make_arg((S, S)))
    yield SampleInput(make_arg((S, S)), make_arg((S, 1)), make_arg((S,)))
    yield SampleInput(make_arg((S,)), make_arg((S, S)), 0.4).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg(()), make_arg((S, S)), 0.4).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, 1)), make_arg((S, S)), 0.4).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, 1)), make_arg((S, S)), make_arg((S, 1))).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, S)), make_arg((S, S)), make_arg((S, S)))
    yield SampleInput(make_arg((S,)), make_arg((S, S)), make_arg((S, S))).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S,)), make_arg((S, S, S)), make_arg((S, S))).with_metadata(broadcasts_input=True)
    yield SampleInput(make_arg((S, S)), make_arg((S, S, S)), make_arg((S,))).with_metadata(broadcasts_input=True)
    if dtype.is_complex:
        yield SampleInput(make_arg((S, S)), make_arg((S, S)), 0.4j)
        yield SampleInput(make_arg((S, S)), make_arg((S, S)), 1.2 + 0.1j)
        yield SampleInput(make_arg((S, S)), make_arg((S,)), 0.4j)
        yield SampleInput(make_arg((S, S)), make_arg((S, S)), 5.4 + 9j)
        yield SampleInput(make_arg(()), make_arg(()), 0.4j)
        yield SampleInput(make_arg(()), make_arg(()), 6.1 + 0.004j)
        yield SampleInput(make_arg((S, S)), make_arg(()), 0.4j)
        yield SampleInput(make_arg((S, S)), make_arg(()), 1 + 2j)

def sample_inputs_tensordot(self, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    cases = (((2, 2, 2), (2, 2, 2), 2), ((2, 2, 1), (2, 1, 2), ([0, 1], [2, 0])))
    for (first_shape, second_shape, dims) in cases:
        yield SampleInput(make_tensor(first_shape, dtype=dtype, device=device, requires_grad=requires_grad), make_tensor(second_shape, dtype=dtype, device=device, requires_grad=requires_grad), dims=dims)

def sample_inputs_kron(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad, low=None, high=None)
    test_cases = (((S, S), (M, L)),)
    for (input_shape, other_shape) in test_cases:
        input = make_arg(input_shape)
        other = make_arg(other_shape)
        yield SampleInput(input, other)

def sample_inputs_inner(self, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(make_arg(S), make_arg(S))
    yield SampleInput(make_arg(), make_arg(S, S))

def sample_inputs_scatter(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')

    def _tensor(shape, dtype=dtype, low=None, high=None):
        if False:
            for i in range(10):
                print('nop')
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        if False:
            return 10
        return gather_variable(shape, index_dim, max_indices, device=device)
    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = ((_tensor((M, S)), (0, _gather((S, S), 1, M), _tensor((S, S)))), (_tensor((M, S)), (1, _gather((S, S), 0, S), _tensor((S, S)))), (_tensor((M, S)), (-1, _gather((S, S), 0, S), _tensor((S, S)))), (_tensor((M, S)), (0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))), (_tensor((M, S)), (1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))), (_tensor((M, S)), (-1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))), (_tensor(()), (0, zero.clone().detach(), _tensor(()))), (_tensor(()), (0, zero.clone().detach(), 2.5)))
    for (tensor, args) in test_cases:
        yield SampleInput(tensor, *args)
        if not requires_grad:
            yield SampleInput(tensor.clone().detach(), *args, reduce='add')
            if dtype.is_floating_point:
                yield SampleInput(tensor.clone().detach(), *args, reduce='multiply')

def sample_inputs_scatter_add(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10

    def _tensor(shape, dtype=dtype, low=None, high=None):
        if False:
            return 10
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)

    def _gather(shape, index_dim, max_indices):
        if False:
            print('Hello World!')
        return gather_variable(shape, index_dim, max_indices, device=device)
    zero = torch.tensor(0, dtype=torch.long, device=device)
    yield SampleInput(_tensor((M, S)), 0, _gather((S, S), 1, M), _tensor((S, S)))
    yield SampleInput(_tensor((M, S)), 1, _gather((S, S), 0, S), _tensor((S, S)))
    yield SampleInput(_tensor((M, S)), -1, _gather((S, S), 0, S), _tensor((S, S)))
    yield SampleInput(_tensor((M, S)), 0, _gather((M, S // 2), 1, M), _tensor((M, S // 2)))
    yield SampleInput(_tensor((M, S)), 1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))
    yield SampleInput(_tensor((M, S)), -1, _gather((M, S // 2), 0, S), _tensor((M, S // 2)))
    yield SampleInput(_tensor(()), 0, zero.clone().detach(), _tensor(()))

def sample_inputs_scatter_reduce(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    gather = partial(gather_variable, device=device)
    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (((M, S), 0, gather((S, S), 1, M), (S, S)), ((M, S), 1, gather((S, S), 0, S), (S, S)), ((M, S), -1, gather((S, S), 0, S), (S, S)), ((M, S), 0, gather((M, S // 2), 1, M), (M, S // 2)), ((M, S), 1, gather((M, S // 2), 0, S), (M, S // 2)), ((M, S), -1, gather((M, S // 2), 0, S), (M, S // 2)), ((), 0, zero.clone().detach(), ()))
    reduce = op_info.variant_test_name
    for ((inp_shape, dim, index, src_shape), include_self) in product(test_cases, [False, True, False]):
        yield SampleInput(make_arg(inp_shape), args=(dim, index, make_arg(src_shape), reduce), kwargs={'include_self': include_self})
    if requires_grad and reduce == 'prod':
        input = torch.tensor([[0, 13], [0, 17], [0, 19]], dtype=dtype, device=device, requires_grad=requires_grad)
        src = torch.tensor([[0, 1, 2, 3], [0, 4, 0, 1], [2, 3, 5, 6]], dtype=dtype, device=device, requires_grad=requires_grad)
        idx = torch.tensor([[1, 1, 0, 0], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=torch.long, device=device)
        yield SampleInput(input, args=(1, idx, src, reduce), kwargs={'include_self': True})

def sample_inputs_segment_reduce(op_info, device, dtype, requires_grad, *, mode='lengths', **kwargs):
    if False:
        return 10

    def _tensor(shape, dtype=dtype, low=None, high=None):
        if False:
            print('Hello World!')
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)
    zero = torch.tensor(0, dtype=torch.long, device=device)
    test_cases = (((S,), 0, [0, 1, 2, 2], False), ((S,), 0, [0, 1, 2, 2], True), ((S,), 0, [2, 0, 3, 0], False), ((S, S), 0, [0, 1, 2, 2], False), ((M, S, S), 0, [1, 2, 0, 6, 0], True), ((S, S), 1, [[0, 1, 2, 2] for _ in range(S)], False), ((S, S), 1, [[2, 0, 3, 0], [0, 1, 2, 2], [3, 0, 2, 0], [1, 1, 1, 2], [0, 1, 2, 2]], False), ((S, S, S), 1, [[0, 1, 2, 2] for _ in range(S)], False), ((S, S, S), 1, [[2, 0, 3, 0], [0, 1, 2, 2], [3, 0, 2, 0], [1, 1, 1, 2], [0, 1, 2, 2]], False))
    reductions = ['max', 'mean', 'min', 'sum', 'prod']
    for (args, reduce, initial) in product(test_cases, reductions, [1, 2]):
        (inp_shape, dim, lengths, unsafe) = args
        lengths_t = torch.tensor(lengths, dtype=torch.long, device=device)
        sample_input_kwargs = {'axis': dim, 'unsafe': unsafe, 'initial': initial}
        if mode == 'lengths':
            sample_input_kwargs['lengths'] = lengths_t
        elif mode == 'offsets':
            zeros_shape = list(lengths_t.shape)
            zeros_shape[dim] = 1
            offsets_t = torch.cat((lengths_t.new_zeros(zeros_shape), lengths_t), dim).cumsum_(dim)
            sample_input_kwargs['offsets'] = offsets_t
        else:
            raise RuntimeError(f"mode most be one of 'offsets' or 'lengths' got '{mode}'.")
        yield SampleInput(_tensor(inp_shape), args=(reduce,), kwargs=sample_input_kwargs)

def sample_inputs_ravel(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, S, S)))
    yield SampleInput(make_arg(()))
    yield SampleInput(make_arg((S, S, S), noncontiguous=True))

def sample_inputs_unravel_index(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, low=None, high=None, requires_grad=requires_grad)
    yield SampleInput(torch.tensor([[3, 8, 13], [0, 5, 10]], device=device, dtype=dtype), (4, 5))
    yield SampleInput(torch.tensor([[3, 8, 13], [0, 5, 10]], device=device, dtype=dtype), (4, 2 ** 30))
    yield SampleInput(torch.tensor([[3, 8, 13], [0, 5, 10]], device=device, dtype=dtype), (2 ** 30, 4))
    yield SampleInput(torch.tensor(2, device=device, dtype=dtype), (2, 2))
    max_val = 2 ** (8 * dtype.itemsize - (1 if dtype.is_signed else 0)) - 1
    yield SampleInput(torch.tensor(max_val - 1, device=device, dtype=dtype), (1, max_val))
    yield SampleInput(torch.tensor([22, 41, 37], device=device, dtype=dtype), (7, 6))
    yield SampleInput(torch.tensor(min(1621, max_val), device=device, dtype=dtype), (6, 7, 8, 9))
    yield SampleInput(torch.tensor([], device=device, dtype=dtype), (10, 3, 5))
    yield SampleInput(torch.tensor([[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0]], device=device, dtype=dtype), (5, 8))
    yield SampleInput(torch.tensor([[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0], [1, 3, 1, 0, 9, 5]], device=device, dtype=dtype), (5, 8, 10))
    yield SampleInput(torch.tensor(0, device=device, dtype=dtype), ())
    a = np.array([[2, 4, 5, 6], [7, 8, 1, 15]])
    b = np.array([[3, 2, 7, 6], [10, 12, 8, 9]])
    (_, i1, i2) = np.intersect1d(a, b, assume_unique=True, return_indices=True)
    yield SampleInput(torch.tensor(i1, device=device, dtype=dtype), a.shape)
    yield SampleInput(torch.tensor(i2, device=device, dtype=dtype), b.shape)
    a = np.array([[2, 4, 5, 6, 6], [4, 7, 8, 7, 2]])
    b = np.array([[3, 2, 7, 7], [10, 12, 8, 7]])
    (_, i1, i2) = np.intersect1d(a, b, return_indices=True)
    yield SampleInput(torch.tensor(i1, device=device, dtype=dtype), a.shape)
    yield SampleInput(torch.tensor(i2, device=device, dtype=dtype), b.shape)

def sample_inputs_tril_triu(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((M, M), ()), ((M, M), (2,)), ((M, S), ()), ((M, S), (-1,)), ((M, M), (2,)), ((S, M, S), ()), ((S, M, S), (2,)), ((3, 3, S, S), ()))
    for (shape, args) in cases:
        yield SampleInput(make_arg(shape), args=args)

def error_inputs_tril_triu(opinfo, device, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_arg((4,))), error_regex='input tensor must have at least 2 dimensions')

def sample_inputs_trilu_indices(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    args_list = ((0, 0), (20, 0), (0, 20), (20, 21, 0), (20, 21, 7), (20, 21, -7))
    for args in args_list:
        yield SampleInput(args[0], args=args[1:], kwargs={'dtype': dtype, 'device': device})

def sample_inputs_clone_contiguous(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(make_arg((S, M, S)))
    yield SampleInput(make_arg(()))

def reference_inputs_clone_contiguous(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_clone_contiguous(op, device, dtype, requires_grad, **kwargs)
    shapes = ((3, 5, 6), (1, 1, 3, 5, 6), (1, 1, 3, 5, 6, 1, 1), (1, 0, 3, 5, 0, 2), (1, 0, 3, 5, 0, 0, 1, 1, 2), ())
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape in shapes:
        yield SampleInput(make_arg(shape))
        yield SampleInput(make_arg(shape).transpose(0, -1))
        yield SampleInput(make_arg(shape, noncontiguous=True))
        yield SampleInput(make_arg(shape, noncontiguous=True).transpose(0, -1))
        yield SampleInput(make_arg(shape), kwargs={'memory_format': torch.contiguous_format})
        yield SampleInput(make_arg(shape).transpose(0, -1), kwargs={'memory_format': torch.contiguous_format})
        yield SampleInput(make_arg(shape, noncontiguous=True), kwargs={'memory_format': torch.contiguous_format})
        yield SampleInput(make_arg(shape, noncontiguous=True).transpose(0, -1), kwargs={'memory_format': torch.contiguous_format})
    strided_cases = (((5, 6, 2), (1, 1, 7), 2), ((5, 5, 4), (1, 1, 7), 2), ((5, 5, 2), (4, 5, 7), 3), ((5, 5, 2), (5, 5, 7), 3), ((5, 5, 2), (5, 5, 5), 3), ((9, 5, 2), (0, 1, 7), 3))
    for (shape, strides, offset) in strided_cases:
        yield SampleInput(make_arg(500).as_strided(shape, strides, offset))
        yield SampleInput(make_arg(500).as_strided(shape, strides, offset), kwargs={'memory_format': torch.contiguous_format})
    yield SampleInput(make_arg((2, 2, 2, 2)), kwargs={'memory_format': torch.channels_last})
    a = make_arg((2, 2, 2, 2)).permute(0, 3, 1, 2)
    yield SampleInput(a, kwargs={'memory_format': torch.channels_last})
    yield SampleInput(make_arg((2, 2, 2, 2, 2)), kwargs={'memory_format': torch.channels_last_3d})
    a = make_arg((2, 2, 2, 2, 2)).permute(0, 4, 1, 2, 3)
    yield SampleInput(a, kwargs={'memory_format': torch.channels_last_3d})

def sample_inputs_sum_to_size(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sample_shapes = [((), ()), ((S,), (1,)), ((S, S), (1, 1)), ((S, S), (1, S)), ((S, S), (S, S)), ((S, S, S), (S, 1, S))]
    for (input_shape, output_shape) in sample_shapes:
        yield SampleInput(make_arg(input_shape), args=(output_shape,))
        if output_shape == ():
            continue
        yield SampleInput(make_arg(input_shape), args=(list(output_shape),))
        yield SampleInput(make_arg(input_shape), args=(*output_shape,))

def error_inputs_sum_to_size(op_info, device, **kwargs):
    if False:
        print('Hello World!')
    shape = (M, S, M)
    err_msg = 'is not expandable to size'
    si = SampleInput(make_tensor(shape, device=device, dtype=torch.float32), args=(M, M))
    yield ErrorInput(si, error_regex=err_msg)
    shape = (M + 1, S, S, M)
    err_msg = 'is not expandable to size'
    si = SampleInput(make_tensor(shape, device=device, dtype=torch.float32), args=(M + 1, 1))
    yield ErrorInput(si, error_regex=err_msg)

def sample_inputs_resize_ops(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    cases = (((S, S, S), (S * S, S)), ((), ()), ((), (1, 1, 1)))
    for (shape, args_or_shape) in cases:
        if op_info.name == 'resize_':
            args = (args_or_shape,)
        elif op_info.name == 'resize_as_':
            args = (make_arg(shape, requires_grad=False),)
        else:
            raise ValueError('sample_inputs_resize_ops is being used with incorrect operator')
        yield SampleInput(make_arg(shape, requires_grad=requires_grad), args=args)

def sample_inputs_view_reshape(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (S * S, S), True), ((S * S, S), (S, S, S), True), ((S * S, S), (S, -1, S), False), ((S * S * 2, S), (S, -1), False), ((S,), (S,), True), ((), (), False), ((), (1,), True))
    for (a, b, is_tensor_supported) in cases:
        if kwargs.get('tensor_arg') and (not is_tensor_supported):
            continue
        if kwargs.get('tensor_arg'):
            b = make_arg(b, requires_grad=False)
        yield SampleInput(make_arg(a), args=(b,))

def reference_inputs_view_reshape(op, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    yield from sample_inputs_view_reshape(op, device, dtype, requires_grad, **kwargs)
    cases = (((125,), (25, 5), True), ((25, 25), (1, 5, 5, 1, 5, 1, 5, 1), True), ((16, 32), (2, 4, 1, 4, 4, 1, 4), True), ((16, 12), (12, 16), True), ((1, 16, 12), (12, 16), True), ((1, 5, 1, 5), (25, 1), True), ((2, 4, 2), (4, 4), True), ((1, 4), (1, 1, 2, 1, 2), True), ((3, 5, 7), (7, 5, 3), True), ((1,), (), False), ((5, 0, 2, 3), (5, 0, 2, 3), True), ((2, 1, 0, 3, 1), (5, 0), True), ((1,), (), False), ((4, 5, 6), (4, 5, 6, 1, 1, 1), True), ((), (1, 1, 1, 1), False))
    irreversible_cases = (((), (-1,), False), ((4, 7, 9, 1, 1), (1, 4, 3, -1, 1), False))
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for (a, b, is_tensor_supported) in cases:
        if kwargs.get('tensor_arg') and (not is_tensor_supported):
            continue
        if kwargs.get('tensor_arg'):
            yield SampleInput(make_arg(a), args=(make_arg(b, requires_grad=False),))
            yield SampleInput(make_arg(b), args=(make_arg(a, requires_grad=False),))
        else:
            yield SampleInput(make_arg(a), args=(b,))
            yield SampleInput(make_arg(b), args=(a,))
    for (a, b, is_tensor_supported) in irreversible_cases:
        if kwargs.get('tensor_arg') and (not is_tensor_supported):
            continue
        if kwargs.get('tensor_arg'):
            b = make_arg(b, requires_grad=False)
        yield SampleInput(make_arg(a), args=(b,))

def error_inputs_view_reshape(op, device, **kwargs):
    if False:
        return 10
    cases = (((2,), (), False), ((1, 3, 0), (), False), ((4, 3), (4, 2), True), ((1, 3, 5), (5, 2, 2), True), ((1, 3, 5), (5, -1, 2), False), ((1, 3, 5), (5, -1, -1), False), (1, (0, -1), False), ((0, 5), (0, -1), False))
    make_arg = partial(make_tensor, dtype=torch.float32, device=device, requires_grad=False)
    for (a, b, is_tensor_supported) in cases:
        if kwargs.get('tensor_arg') and (not is_tensor_supported):
            continue
        if b == (5, -1, -1):
            error_regex = 'only one dimension can be inferred'
        elif a == (0, 5):
            error_regex = 'cannot reshape tensor of 0 elements into shape \\[0, -1\\] because the unspecified dimension size -1 can be any value and is ambiguous'
        else:
            shape = ', '.join(map(str, b))
            size = a if type(a) is int else functools.reduce(operator.mul, a, 1)
            error_regex = f"shape '\\[{shape}\\]' is invalid for input of size {size}"
        if kwargs.get('tensor_arg'):
            b = make_arg(b, requires_grad=False)
        yield ErrorInput(SampleInput(make_arg(a), args=(b,)), error_type=Exception, error_regex=error_regex)

def sample_inputs_atleast1d2d3d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    input_list = []
    shapes = ((S, S, S, S), (S, S, S), (S, S), (S,), ())
    make_tensor_partial = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape in shapes:
        yield SampleInput(make_tensor_partial(shape))
    yield SampleInput([make_tensor_partial(shape) for shape in shapes])

def sample_inputs_column_stack(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    cases: Tuple[tuple, tuple] = (((S, 2, 1), (S, 3, 1)), (S, (S, 5)), ((), (1, S)))
    make_tensor_partial = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for (shape1, shape2) in cases:
        yield SampleInput([make_tensor_partial(shape1), make_tensor_partial(shape2)])

def sample_inputs_flatten(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    shapes = ((S, S, S), (S, S), (S,), ())
    make_tensor_partial = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for shape in shapes:
        yield SampleInput(make_tensor_partial(shape))
        if len(shape) > 1:
            yield SampleInput(make_tensor_partial(shape), start_dim=1, end_dim=-1)

def reference_inputs_flatten(op, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_flatten(op, device, dtype, requires_grad, **kwargs)
    cases = (((5, 4, 0, 1, 3, 7), 1, 3), ((5, 4, 0, 1, 3, 7), 4, 5), ((5, 4, 1, 1, 3, 7), 2, 3), ((), 0, -1), ((1,), 0, -1), ((3, 7, 5), 1, 2), ((4, 5), 1, 1), ((1, 5, 5, 1, 5, 1, 5, 1), 0, 2), ((1, 5, 5, 1, 5, 1, 5, 1), 3, -1), ((1, 5, 5, 1, 5, 7, 5, 1), -2, -1), ((2, 4, 2), 0, 1), ((4, 2, 2), 1, 2), ((0, 3, 4, 5), 1, 3))
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for (shape, start, end) in cases:
        yield SampleInput(make_arg(shape), args=(start, end))
        yield SampleInput(make_arg(shape, noncontiguous=True).transpose(0, -1), args=(start, end))
        yield SampleInput(make_arg(shape).transpose(0, -1), args=(start, end))

def sample_inputs_unflatten(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    args = (((8,), 0, (8,)), ((8,), 0, (4, 2)), ((8,), -1, (2, 2, 2)), ((8,), -1, (-1, 2)), ((3, 6, 2), 1, (2, 3)), ((3, 6, 2), -2, (2, 3)), ((3, 6, 2), -2, (-1, 3)), ((3, 2, 12), 2, (3, 2, 2)), ((4, 0), 0, (2, 2)), ((4, 0), 1, (2, 0, 0, 0)))
    make_tensor_partial = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    for (in_shape, dim, sizes) in args:
        yield SampleInput(make_tensor_partial(in_shape), args=(dim, sizes))

def sample_inputs_select(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (1, 2)), ((S, S, S), (-1, 2)), ((S, S, S), (-1, -1)), ((S, S, S), (1, -1)), ((S,), (0, 2)))
    for (shape, args) in cases:
        yield SampleInput(make_arg(shape), args=args)

def sample_inputs_select_scatter(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (S, S), (1, 2)), ((S, S, S), (S, S), (-1, 2)), ((S, S, S), (S, S), (-1, -1)), ((S, S, S), (S, S), (1, -1)), ((S,), (), (0, 2)))
    for (input_shape, src_shape, args) in cases:
        input_ = make_arg(input_shape)
        src = make_arg(src_shape)
        yield SampleInput(input_, args=(src, *args))

def sample_inputs_slice_scatter(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((L, L, L), (L, L, L), (0, 0, L, 1)), ((L, L, L), (L // 2, L, L), (0, L // 2, L, 1)), ((L, L, L), (L // 4, L, L), (0, L // 2, L, 2)), ((L, L, L), (L, L, L), (1, 0, L, 1)), ((L, L, L), (L, L // 2, L), (1, L // 2, L, 1)), ((L, L, L), (L, L // 4, L), (1, L // 2, L, 2)), ((L, L, L), (L, L, L), (2, 0, L, 1)), ((L, L, L), (L, L, L // 2), (2, L // 2, L, 1)), ((L, L, L), (L, L, L // 4), (2, L // 2, L, 2)))
    for (input_shape, src_shape, args) in cases:
        input_ = make_arg(input_shape)
        src = make_arg(src_shape)
        yield SampleInput(input_, args=(src, *args))

def sample_inputs_expand(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, 1, 1), (S, S, S)), ((S, 1, S), (S, S, S)), ((S, 1, S), (-1, S, -1)), ((S, 1, S), (-1, S, S)), ((S, 1), (S, S, S)), ((1,), (S, S, S)), ((1, S), (1, 1, S)), ((), ()), ((), (1, 3, 2)))
    for case in cases:
        (shape, args) = case
        yield SampleInput(make_arg(shape), args=(args,))

def sample_inputs_conversion(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    shapes = ((), (2, 3))
    memory_format_options = [None, torch.contiguous_format]
    for (shape, memory_format) in itertools.product(shapes, memory_format_options):
        yield SampleInput(make_arg(shape), kwargs={'memory_format': memory_format} if memory_format else {})
    yield SampleInput(make_arg((2, 3, 2, 3)), kwargs={'memory_format': torch.channels_last})

def sample_inputs_expand_as(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, dtype=dtype, device=device)
    cases = (((S, 1, 1), (S, S, S)), ((), ()), ((), (1, 1)))
    for (shape, shape_other) in cases:
        yield SampleInput(make_arg(shape, requires_grad=requires_grad), args=(make_arg(shape_other, requires_grad=False),))

def sample_inputs_where(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)

    def make_bool_mask(shape):
        if False:
            i = 10
            return i + 15
        mask_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)
        if mask_t.numel() == 0:
            return mask_t
        elif mask_t.numel() == 1:
            mask_t.fill_(True)
            return mask_t
        if mask_t.sum() == 0:

            def random_index(shape):
                if False:
                    for i in range(10):
                        print('nop')
                return tuple((random.randrange(0, max_idx) for max_idx in shape))
            mask_t[random_index(mask_t.shape)] = True
            return mask_t
        return mask_t
    cases = (((M, M), (M, M), (M, M), False), ((M, 1, M), (M, M), (M, M, 1), True), ((), (), (), False), ((M, 1, M), (), (M, M, 1), True), ((), (M, M), (), True), ((), 2, (1, 1), True))
    for (shape, mask_shape, other_shape, broadcasts_input) in cases:
        yield SampleInput(make_arg(shape), args=(make_bool_mask(mask_shape), make_arg(other_shape)), broadcasts_input=broadcasts_input)

def reference_inputs_where(op, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    yield from sample_inputs_where(op, device, dtype, requires_grad, **kwargs)
    make_cond = partial(make_tensor, dtype=torch.bool, device=device, requires_grad=requires_grad)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    c = make_cond((10, 3), noncontiguous=True)
    a = make_arg((10, 1), noncontiguous=True)
    b = make_arg((3, 10, 3)).transpose(0, -1)
    yield SampleInput(a, args=(c, b))
    other_dtype = torch.double if dtype is not torch.double else torch.long
    c = make_cond((10, 3), noncontiguous=True)
    a = make_arg((10, 1), dtype=torch.long)
    b = make_arg((10, 1))
    yield SampleInput(a, args=(c, b))
    c = make_cond((10, 3), noncontiguous=True)
    a = make_arg((1,)).item()
    b = make_arg((1,)).item()
    yield SampleInput(a, args=(c, b))
    if dtype.is_floating_point or dtype.is_complex:
        if dtype.is_floating_point:
            nan = float('nan')
        else:
            nan = complex(float('nan'), float('nan'))
        c = make_cond((1, 10, 3))
        a = make_arg((10, 3), noncontiguous=True)
        a[2, 1] = nan
        b = make_arg((1, 3))
        b[0, 2] = nan
        yield SampleInput(a, args=(c, b))
    for scalar in (0, 0.0, 2j, False):
        yield SampleInput(scalar, args=(c, b))
        yield SampleInput(a, args=(c, scalar))

def error_inputs_where(op_info, device, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    shape = (S,)
    err_msg = 'Expected all tensors to be on the same device'
    for devices in product(('cpu', device), repeat=3):
        if len(set(devices)) == 2:
            si = SampleInput(make_tensor(shape, device=devices[0], dtype=torch.float32), args=(make_tensor(shape, dtype=torch.bool, device=devices[1]), make_tensor(shape, device=devices[2], dtype=torch.float32)))
            yield ErrorInput(si, error_regex=err_msg)

def sample_inputs_nonzero(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))
    inputs = []
    for shape in sizes:
        zeros = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        inputs.append(zeros)
        mixed = make_arg(shape).requires_grad_(False)
        mask_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)
        mixed[mask_t] = 0
        inputs.append(mixed)
    for (input_t, as_tuple) in product(inputs, [False, True]):
        yield SampleInput(input_t.clone().requires_grad_(requires_grad), kwargs=dict(as_tuple=as_tuple))

def sample_inputs_nonzero_static(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    sizes = ((), (S,), (S, S), (S, S, S), (S, 1, S), (S, 0, S))
    inputs = []
    for shape in sizes:
        zeros = torch.zeros(shape, dtype=dtype, device=device, requires_grad=requires_grad)
        inputs.append(zeros)
        mixed = make_arg(shape).requires_grad_(False)
        mask_t = make_tensor(shape, dtype=torch.bool, device=device, requires_grad=False)
        mixed[mask_t] = 0
        inputs.append(mixed)
    nonzero_sizes = [0, 1, XS, S, M]
    for (input_t, nonzero_size) in product(inputs, nonzero_sizes):
        yield SampleInput(input_t.clone().requires_grad_(requires_grad), kwargs=dict(size=nonzero_size))

def sample_inputs_chunk(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((S, S, S), (2,)), ((S, S, S), (S, 1)), ((S, S, S), (S, -1)))
    for case in cases:
        (shape, args) = case
        yield SampleInput(make_arg(shape), args=args)

def reference_inputs_chunk(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_chunk(op, device, dtype, requires_grad, **kwargs)
    make_arg = partial(make_tensor, dtype=dtype, device=device, requires_grad=requires_grad)
    cases = (((13, 9, 11), 17, -1), ((13, 9, 11), 11, -1), ((13,), 12, -1), ((15,), 12, -1), ((15,), 7, 0), ((15,), 9, 0), ((3, 7), 9, 1), ((3, 7), 9, 0), ((3, 7), 2, 0), ((3, 7), 3, 0), ((3, 7), 1, 0), ((3, 7), 1, 1), ((4, 4), 2, 0))
    for (shape, chunks, dim) in cases:
        yield SampleInput(make_arg(shape), args=(chunks, dim))

def sample_inputs_kthvalue(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10

    def _tensor(shape, dtype=dtype, low=None, high=None):
        if False:
            print('Hello World!')
        return make_tensor(shape, dtype=dtype, device=device, low=low, high=high, requires_grad=requires_grad)
    test_cases = [((S, S, S), (2,)), ((S, S, S), (2, 1)), ((S, S, S), (2, -1)), ((S, S, S), (2, 1, True)), ((S, S, S), (2, -1, True)), ((S,), (2, 0)), ((S,), (2, 0, True)), ((), (1,)), ((), (1, 0)), ((), (1, 0, True))]
    yield from (SampleInput(_tensor(tensor), *args) for (tensor, args) in test_cases)

def error_inputs_kthvalue(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    t = make_tensor(10, dtype=torch.float32, device=device)
    indices = torch.empty((), device=device, dtype=torch.long)
    yield ErrorInput(SampleInput(t, 5, out=(t, indices)), error_regex='unsupported operation')
    k_out_of_range_err = 'selected number k out of range for dimension'
    yield ErrorInput(SampleInput(torch.randn(2, 2, device=device), 3, 0), error_regex=k_out_of_range_err)
    yield ErrorInput(SampleInput(torch.randn(2, 2, device=device), 3), error_regex=k_out_of_range_err)
    yield ErrorInput(SampleInput(torch.tensor(2, device=device), 3), error_regex=k_out_of_range_err)

def sample_inputs_dropout(op_info, device, dtype, requires_grad, *, train=None, valid_input_dim=None, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    if valid_input_dim:
        cases = ((S,) * i for i in valid_input_dim)
    else:
        cases = ((S, S), (S,), ())
    p_vals = [0.0, 0.5, 1.0]
    training_vals = [train] if train is not None else [True, False]
    for (case, p, training) in product(cases, p_vals, training_vals):
        yield SampleInput(make_arg(case), p=p, training=training)
    yield SampleInput(make_arg(case))

def sample_inputs_dropout_backward(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_mask = partial(make_tensor, device=device, dtype=torch.bool, requires_grad=False)
    cases = ((S, S, S, S), (S,), ())
    scale_vals = [0.0, 1.0, 2.0]
    for (case, scale) in product(cases, scale_vals):
        yield SampleInput(make_arg(case), make_mask(case), scale)

def sample_inputs_embedding_bag(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')

    def make_input(shape):
        if False:
            for i in range(10):
                print('nop')
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_long_input(shape, *, low, high, noncontiguous=False):
        if False:
            print('Hello World!')
        return make_tensor(shape, device=device, dtype=torch.long, low=low, high=high, noncontiguous=noncontiguous)

    def make_per_sample_weight(flag, idx):
        if False:
            i = 10
            return i + 15
        if flag:
            return make_input(idx.shape)
        return None
    offsets = torch.tensor([0, 3], device=device, dtype=torch.long)
    for generate_per_sample_weight in (True, False):
        for mode in ('sum', 'mean', 'max'):
            if generate_per_sample_weight and mode in ('mean', 'max'):
                continue
            idx = make_long_input((S,), low=0, high=M)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,), kwargs={'offsets': offsets, 'mode': mode, 'per_sample_weights': per_sample_weights})
            idx = make_long_input((S,), low=0, high=M, noncontiguous=True)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,), kwargs={'offsets': offsets, 'mode': mode, 'per_sample_weights': per_sample_weights})
            idx = make_long_input((S,), low=0, high=M, noncontiguous=True)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,), kwargs={'offsets': torch.tensor([0, 0, 3], device=device, dtype=torch.long), 'mode': mode, 'per_sample_weights': per_sample_weights})
            idx = make_long_input((S, S), low=0, high=M)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,), kwargs={'mode': mode, 'per_sample_weights': per_sample_weights})
            idx = make_long_input((S, S), low=0, high=M, noncontiguous=True)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((M, S)), args=(idx,), kwargs={'mode': mode, 'per_sample_weights': per_sample_weights})
            idx = make_long_input((6,), low=0, high=S)
            idx[0] = 4
            idx[4] = 4
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': -1, 'offsets': offsets, 'mode': mode, 'per_sample_weights': per_sample_weights})
            idx = make_long_input((3, 3), low=0, high=S)
            idx[0, 0] = 2
            idx[1, 1] = 2
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': 2, 'mode': mode, 'per_sample_weights': per_sample_weights})
            idx = make_long_input((6,), low=0, high=S)
            weights = make_input((S, S))
            offsets_ = torch.tensor([0, 3, 6], device=device, dtype=torch.long)
            per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
            yield SampleInput(weights, args=(idx,), kwargs={'mode': mode, 'offsets': offsets_, 'include_last_offset': True})
            if not requires_grad:
                idx = make_long_input((2, 2), low=0, high=S)
                weights = make_input((S, S)) * 2
                per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1.0, 'mode': mode, 'per_sample_weights': per_sample_weights})
                idx = make_long_input((6,), low=0, high=S)
                weights = make_input((S, S)) * 2
                per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1.0, 'norm_type': 1.0, 'mode': mode, 'offsets': offsets, 'per_sample_weights': per_sample_weights})
                if mode != 'max':
                    idx = make_long_input((2, 2), low=0, high=S)
                    idx[0, 0] = 1
                    idx[0, 1] = 1
                    weights = make_input((S, S))
                    per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                    yield SampleInput(weights, args=(idx,), kwargs={'scale_grad_by_freq': True, 'mode': mode, 'per_sample_weights': per_sample_weights})
                    idx = make_long_input((6,), low=0, high=S)
                    weights = make_input((S, S))
                    per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                    yield SampleInput(weights, args=(idx,), kwargs={'sparse': True, 'offsets': offsets, 'mode': mode, 'per_sample_weights': per_sample_weights})
                    idx = make_long_input((6,), low=0, high=S)
                    idx[0] = 1
                    idx[1] = 1
                    idx[3] = 0
                    weights = make_input((S, S)) * 2
                    per_sample_weights = make_per_sample_weight(generate_per_sample_weight, idx)
                    yield SampleInput(weights, args=(idx,), kwargs={'sparse': True, 'scale_grad_by_freq': True, 'padding_idx': 0, 'max_norm': 1.0, 'offsets': offsets, 'mode': mode, 'per_sample_weights': per_sample_weights})

def sample_inputs_embedding(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10

    def make_input(shape):
        if False:
            return 10
        return make_tensor(shape, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_long_input(shape, *, low, high):
        if False:
            i = 10
            return i + 15
        return make_tensor(shape, device=device, dtype=torch.long, low=low, high=high)
    idx = make_long_input((), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,))
    idx = make_long_input((S,), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,))
    idx = make_long_input((S, S), low=0, high=M)
    yield SampleInput(make_input((M, S)), args=(idx,))
    if not requires_grad:
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 2
        idx[1, 1] = 2
        yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': 2})
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 4
        idx[1, 1] = 4
        yield SampleInput(make_input((S, S)), args=(idx,), kwargs={'padding_idx': -1})
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1.0})
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'max_norm': 1.0, 'norm_type': 1.0})
        idx = make_long_input((2, 2), low=0, high=S)
        idx[0, 0] = 1
        idx[0, 1] = 1
        weights = make_input((S, S))
        yield SampleInput(weights, args=(idx,), kwargs={'scale_grad_by_freq': True})
        idx = make_long_input((2, 2), low=0, high=S)
        weights = make_input((S, S))
        yield SampleInput(weights, args=(idx,), kwargs={'sparse': True})
        idx = make_long_input((3, 3), low=0, high=S)
        idx[0, 0] = 1
        idx[0, 1] = 1
        idx[1, 0] = 0
        weights = make_input((S, S)) * 2
        yield SampleInput(weights, args=(idx,), kwargs={'sparse': True, 'scale_grad_by_freq': True, 'padding_idx': 0, 'max_norm': 1.0})

def sample_inputs_one_hot(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15

    def make_input(shape, *, low, high):
        if False:
            return 10
        return make_tensor(shape, device=device, dtype=dtype, low=low, high=high, requires_grad=requires_grad)
    shapes = ((), (S,), (L, M, S))
    num_classess = (-1, 10)
    return (SampleInput(make_input(shape, low=0, high=10 if num_classes == -1 else num_classes // 2), kwargs=dict(num_classes=num_classes)) for (shape, num_classes) in itertools.product(shapes, num_classess))

def sample_inputs_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    rhs_requires_grad = kwargs.get('rhs_requires_grad', requires_grad)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shapes_and_kwargs = (((), None), ((S,), dict(reduction='mean')), ((S,), dict(reduction='sum')), ((S,), dict(reduction='none')), ((S, S), None), ((S, S, S), None))
    for (shape, kwargs) in shapes_and_kwargs:
        yield SampleInput(_make_tensor(shape), args=(_make_tensor(shape, requires_grad=rhs_requires_grad),), kwargs=kwargs)

def sample_inputs_grid_sample(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-2, high=2)
    batch_size = 2
    num_channels = 3
    modes = ('bilinear', 'nearest')
    align_cornerss = (False, True)
    padding_modes = ('zeros', 'border', 'reflection')
    for dim in (2, 3):
        modes_ = (*modes, 'bicubic') if dim == 2 else modes
        for (mode, padding_mode, align_corners) in itertools.product(modes_, padding_modes, align_cornerss):
            yield SampleInput(_make_tensor((batch_size, num_channels, *[S] * dim)), _make_tensor((batch_size, *[S] * dim, dim)), mode=mode, padding_mode=padding_mode, align_corners=align_corners)

def reference_inputs_grid_sample(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    batch_size = 2
    num_channels = 3
    height = 345
    width = 456
    modes = ('bilinear', 'nearest', 'bicubic')
    align_cornerss = (False, True)
    padding_modes = ('zeros', 'border', 'reflection')
    a = torch.deg2rad(torch.tensor(45.0))
    (ca, sa) = (torch.cos(a), torch.sin(a))
    (s1, s2) = (1.23, 1.34)
    theta = torch.tensor([[[ca / s1, sa, 0.0], [-sa, ca / s2, 0.0]]], dtype=dtype, device=device)
    theta = theta.expand(batch_size, 2, 3).contiguous()
    x = torch.arange(batch_size * num_channels * height * width, device=device)
    x = x.reshape(batch_size, num_channels, height, width).to(torch.uint8)
    x = x.to(dtype=dtype)
    x.requires_grad_(requires_grad)
    for (mode, padding_mode, align_corners) in itertools.product(modes, padding_modes, align_cornerss):
        grid = torch.nn.functional.affine_grid(theta, size=(batch_size, num_channels, height, width), align_corners=align_corners)
        yield SampleInput(x, grid, mode, padding_mode, align_corners)

def sample_inputs_grid_sampler_2d(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad, low=-2, high=2)
    batch_size = 2
    num_channels = 3
    modes = (0, 1, 2)
    align_cornerss = (False, True)
    padding_modes = (0, 1, 2)
    for (mode, padding_mode, align_corners) in itertools.product(modes, padding_modes, align_cornerss):
        yield SampleInput(_make_tensor((batch_size, num_channels, S, L)), _make_tensor((batch_size, M + 3, M, 2)), mode, padding_mode, align_corners)

def sample_inputs_cosine_embedding_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_target(shape):
        if False:
            for i in range(10):
                print('nop')
        shape = () if len(shape) == 1 else (shape[0],)
        t = torch.randint(0, 2, shape, device=device, dtype=torch.long)
        t = t * 2 - 1
        target = t.to(dtype=dtype).detach_().requires_grad_(requires_grad)
        return target
    shapes = ((S, S), (S,))
    reductions = ('none', 'mean', 'sum')
    for (s, r) in product(shapes, reductions):
        yield SampleInput(make_input(s), args=(make_input(s), make_target(s)), kwargs=dict(reduction=r, margin=random.uniform(-1, 1)))

def sample_inputs_ctc_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    input_length = 50
    batch = 16
    num_char = 20
    target_length = 30

    def make_log_probs(s):
        if False:
            i = 10
            return i + 15
        t = make_tensor(s, device=device, dtype=dtype)
        log_probs = t.log_softmax(2).to(device=device, dtype=dtype).detach().requires_grad_(requires_grad=requires_grad)
        return log_probs
    reductions = ('none', 'mean', 'sum')
    zero_inf = (True, False)
    lengths_type = (list, torch.Tensor)
    for (r, z, lt) in product(reductions, zero_inf, lengths_type):
        log_probs = make_log_probs((input_length, batch, num_char))
        targets = torch.randint(1, num_char, (batch, target_length), dtype=torch.long, device=device)
        input_lengths = torch.full((batch,), input_length, dtype=torch.long, device=device)
        target_lengths = torch.randint(10, target_length, (batch,), dtype=torch.long, device=device)
        if lt is list and r in ['none', 'sum']:
            input_lengths = input_lengths.tolist()
            target_lengths = target_lengths.tolist()
        yield SampleInput(log_probs, args=(targets, input_lengths, target_lengths), kwargs=dict(reduction=r, zero_infinity=z))

def sample_inputs_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    shape = (2, 3)
    num_classes = shape[1]
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_weight = partial(make_tensor, num_classes, device=device, dtype=dtype, requires_grad=False)

    def make_target(shape, zeros=False):
        if False:
            return 10
        s = (shape[0], *shape[2:]) if len(shape) > 1 else ()
        if zeros:
            return torch.zeros(s, device=device, dtype=torch.long)
        else:
            return make_tensor(s, low=0, high=shape[1] if len(shape) > 1 else shape[0], device=device, dtype=torch.long)

    def gen_shape_kwargs():
        if False:
            return 10
        shapes = (shape, (num_classes,), shape + (2, 2))
        reductions = ('none', 'mean', 'sum')
        for (reduction, s) in product(reductions, shapes):
            yield (make_input(s), make_target(s), dict(reduction=reduction))
            yield (make_input(s), make_target(s), dict(weight=make_weight(), reduction=reduction))
            yield (make_input(s), make_target(s), dict(weight=make_weight(low=0), reduction=reduction))
            yield (make_input(s), make_target(s), dict(weight=make_weight(high=0), reduction=reduction))
            t = make_target(s)
            ignore = num_classes // 2
            if t.eq(ignore).all() and reduction == 'mean':
                t.fill_(0)
            yield (make_input(s), t, dict(ignore_index=num_classes // 2, reduction=reduction))
            yield (make_input(s), t, dict(ignore_index=num_classes // 2, reduction=reduction, weight=make_weight()))
            if reduction != 'mean':
                yield (make_input(s), make_target(s, zeros=True), dict(ignore_index=0, reduction=reduction))
    for (input, target, kwargs) in gen_shape_kwargs():
        yield SampleInput(input, args=(target,), kwargs=kwargs)
    target = torch.tensor([-1, 2], device=device, dtype=torch.long)
    yield SampleInput(make_input(shape), args=(target,), kwargs={'ignore_index': -1})

def sample_inputs_binary_cross_entropy_with_logits(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make = partial(make_tensor, device=device, dtype=dtype)
    make_prob = partial(make, low=0, high=1)
    reductions = ('mean', 'sum', 'none')

    def make_weight_shape_kwargs():
        if False:
            while True:
                i = 10
        kwargs = []
        for shape in ((1,), (1, S), S, (S, S)):
            kwargs.extend([((S, S), dict(reduction=reduction, weight=make(shape))) for reduction in reductions])
        return kwargs
    shapes_and_kwargs = [*[(shape, None) for shape in ((), (1,), (S,), (S, S), (S, S, S))], *[((S, S), dict(reduction=reduction)) for reduction in reductions], *make_weight_shape_kwargs(), *[((S, S), dict(reduction=reduction, pos_weight=make((S,), low=0))) for reduction in reductions], *[((S, S), dict(reduction=reduction, weight=make((S, S)), pos_weight=make((S,), low=0))) for reduction in reductions]]
    for (shape, kwargs) in shapes_and_kwargs:
        yield SampleInput(make(shape, requires_grad=requires_grad), args=(make_prob(shape, requires_grad=requires_grad),), kwargs=kwargs)

def sample_inputs_argwhere(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    yield SampleInput(torch.tensor([1, 0, 2, 0], dtype=dtype, device=device, requires_grad=requires_grad))
    mask = torch.tensor([[0, 1, 0, 1, 0], [1, 1, 1, 1, 0], [0, 0, 0, 1, 0], [1, 0, 1, 1, 0], [1, 0, 0, 1, 0]], dtype=torch.bool, device=device)
    t = make_tensor((S, S), dtype=dtype, device=device, requires_grad=requires_grad)
    t[mask] = 0
    yield SampleInput(t)
    t = make_tensor((S, S), dtype=dtype, device=device, requires_grad=requires_grad, noncontiguous=True)
    t[mask] = 0
    yield SampleInput(t)
    t = make_tensor((S, 0), dtype=dtype, device=device, requires_grad=requires_grad)
    yield SampleInput(t)
    yield SampleInput(torch.zeros((S,), dtype=dtype, device=device, requires_grad=requires_grad))
    yield SampleInput(make_tensor((), dtype=dtype, device=device, requires_grad=requires_grad))

def _generate_sample_shape_reduction():
    if False:
        for i in range(10):
            print('nop')
    shapes = ((S,), (S, S), (S, S, S))
    reductions = ('none', 'mean', 'sum')
    yield from product(shapes, reductions)

def sample_inputs_gaussian_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    make_var = partial(make_tensor, low=0.1, device=device, dtype=dtype, requires_grad=requires_grad)

    def gen_shape(shape):
        if False:
            return 10
        yield shape
        yield (*shape[:-1], 1)
        yield shape[:-1]

    def gen_shape_kwargs():
        if False:
            print('Hello World!')
        for (s, r) in _generate_sample_shape_reduction():
            for (t_s, v_s) in product(gen_shape(s), gen_shape(s)):
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(reduction=r))
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(full=True, reduction=r))
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(eps=random.uniform(1e-06, 0.001), reduction=r))
                yield (_make_tensor(s), _make_tensor(t_s), make_var(v_s), dict(full=True, eps=random.uniform(1e-06, 0.001), reduction=r))
    for (input, target, var, kwargs) in gen_shape_kwargs():
        yield SampleInput(input, args=(target, var), kwargs=kwargs)

def error_inputs_gaussian_nll_loss(op_info, device, **kwargs):
    if False:
        print('Hello World!')
    _make = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(_make(10, 2, 3), _make(10, 2, 3), _make((10, 2, 3), low=0), reduction='abc'), error_type=ValueError, error_regex='abc is not valid')
    yield ErrorInput(SampleInput(_make(10, 2, 3), _make(10, 2, 3), _make((10, 2, 2), low=0)), error_type=ValueError, error_regex='var is of incorrect size')
    yield ErrorInput(SampleInput(_make(10, 2, 3), _make(10, 2, 2), _make((10, 2, 3), low=0)), error_type=RuntimeError, error_regex='The size of tensor a \\(3\\) must match the size of tensor b \\(2\\) at non-singleton dimension 2')

def _generate_sample_inputs_nn_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for (s, r) in _generate_sample_shape_reduction():
        yield (_make_tensor(s), _make_tensor(s), dict(reduction=r))

def sample_inputs_hinge_embedding_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    for (input, target, d) in _generate_sample_inputs_nn_loss(op_info, device, dtype, requires_grad, **kwargs):
        mask = torch.rand_like(target) > 0.5
        target[mask] = 1
        target[~mask] = -1
        d['margin'] = random.uniform(-9, 9)
        yield SampleInput(input, args=(target,), kwargs=d)
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(_make_tensor(()), args=(_make_tensor(()),))

def error_inputs_hinge_embedding_loss(op, device, **kwargs):
    if False:
        return 10
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='is not a valid value')

def reference_inputs_hinge_embedding_loss(op, device, dtype, requires_grad, **kwargs):
    if False:
        return 10
    yield from sample_inputs_hinge_embedding_loss(op, device, dtype, requires_grad, **kwargs)
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    for reduction in ('sum', 'mean', 'none'):
        if dtype.is_floating_point:
            inp = make_input((10,))
            inp[2] = float('nan')
            target = make_input((10,))
            mask = torch.rand_like(target) > 0.5
            target[mask] = -1
            target[~mask] = 1
            yield SampleInput(inp, args=(target,), kwargs={'reduction': reduction})
            inp = make_input((10,))
            inp[4] = float('inf')
            target = make_input((10,))
            mask = torch.rand_like(target) > 0.5
            target[mask] = -1
            target[~mask] = 1
            yield SampleInput(inp, args=(target,), kwargs={'reduction': reduction})
        inp = make_input((5, 5))
        target = make_input((1, 5))
        mask = torch.rand_like(target) > 0.5
        target[mask] = -1
        target[~mask] = 1
        yield SampleInput(inp, args=(target,), kwargs={'reduction': reduction})

def sample_inputs_huber_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    for (input, target, d) in _generate_sample_inputs_nn_loss(op_info, device, dtype, requires_grad, **kwargs):
        d['delta'] = random.uniform(0.001, 9)
        yield SampleInput(input, args=(target,), kwargs=d)

def error_inputs_huber_loss(op, device, **kwargs):
    if False:
        print('Hello World!')
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    err = 'is not a valid value for reduction'
    yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex=err)
    for delta in (0, -1):
        err = 'huber_loss does not support non-positive values for delta.'
        yield ErrorInput(SampleInput(make_input(5, 4), args=(make_input(5, 4),), kwargs={'delta': delta}), error_type=RuntimeError, error_regex=err)

def sample_inputs_poisson_nll_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    _make_tensor = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)

    def gen_shape_kwargs():
        if False:
            return 10
        for (s, r) in _generate_sample_shape_reduction():
            for li in (True, False):
                for f in (True, False):
                    i1 = _make_tensor(s)
                    i2 = _make_tensor(s)
                    t1 = _make_tensor(s, low=0)
                    t2 = _make_tensor(s, low=0)
                    if not li:
                        i1.abs_()
                        i2.abs_()
                    t1.abs_()
                    t2.abs_()
                    yield (i1, t1, dict(log_input=li, full=f, reduction=r))
                    yield (i2, t2, dict(log_input=li, full=f, eps=random.uniform(1e-08, 0.001), reduction=r))
    for (input, target, kwargs) in gen_shape_kwargs():
        yield SampleInput(input, args=(target,), kwargs=kwargs)
    if dtype.is_complex:
        for d in (torch.bool, torch.int64):
            yield SampleInput(_make_tensor(dtype=dtype), args=(_make_tensor(dtype=d),))
            yield SampleInput(_make_tensor(dtype=d), args=(_make_tensor(dtype=dtype),))

def error_inputs_poisson_nll_loss(op_info, device, **kwargs):
    if False:
        return 10
    make = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5, 4),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='abc is not a valid value for reduction')
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5),)), error_regex='(Attempting to broadcast a dimension of length|The size of tensor a \\(5\\) must match the size of tensor b \\(4\\) at non-singleton dimension 1)')

def error_inputs_soft_margin_loss(op_info, device, **kwargs):
    if False:
        while True:
            i = 10
    make = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5, 4),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='abc is not a valid value for reduction')
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5),)), error_regex='(Attempting to broadcast a dimension of length|The size of tensor a \\(4\\) must match the size of tensor b \\(5\\) at non-singleton dimension 1)')

def sample_inputs_triplet_margin_loss(op_info, device, dtype, requires_grad, with_distance=False, **kwargs):
    if False:
        while True:
            i = 10
    make = partial(make_tensor, (S, M), device=device, dtype=dtype, requires_grad=requires_grad)
    kwargss = (*[dict(margin=margin) for margin in (1e-06, 1.0, 10.0)], dict(swap=True), *[dict(reduction=reduction) for reduction in ('mean', 'sum', 'none')])
    for kwargs in kwargss:
        input = make()
        args = (make(), make())
        if with_distance:
            kwargs['distance_function'] = torch.nn.PairwiseDistance()
        yield SampleInput(input, args=args, kwargs=kwargs)

def error_inputs_triplet_margin_loss(op_info, device, **kwargs):
    if False:
        i = 10
        return i + 15
    make_input = partial(make_tensor, device=device, dtype=torch.float32)
    samples = ((make_input(3, 4), (make_input(3, 4), make_input(3, 4)), dict(reduction='abc'), ValueError, 'abc is not a valid value for reduction'), (make_input(3, 5), (make_input(3, 4), make_input(3, 4)), dict(), RuntimeError, '(Attempting to broadcast a dimension of length|The size of tensor a \\(5\\) must match the size of tensor b \\(4\\) at non-singleton dimension 1)'), (make_input(3, 4), (make_input(3, 5), make_input(3, 4)), dict(), RuntimeError, '(Attempting to broadcast a dimension of length|The size of tensor a \\(4\\) must match the size of tensor b \\(5\\) at non-singleton dimension 1)'), (make_input(3, 4), (make_input(3, 4), make_input(3, 5)), dict(), RuntimeError, '(Attempting to broadcast a dimension of length|The size of tensor a \\(4\\) must match the size of tensor b \\(5\\) at non-singleton dimension 1)'), (make_input(3), (make_input(3, 4), make_input(3, 4)), dict(), RuntimeError, 'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor 1D, positive 2D, and negative 2D inputs'), (make_input(3, 4), (make_input(3), make_input(3, 4)), dict(), RuntimeError, 'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor 2D, positive 1D, and negative 2D inputs'), (make_input(3, 4), (make_input(3, 4), make_input(3)), dict(), RuntimeError, 'The anchor, positive, and negative tensors are expected to have the same number of dimensions, but got: anchor 2D, positive 2D, and negative 1D inputs'))
    for (input, args, kwargs, error_type, error_regex) in samples:
        yield ErrorInput(SampleInput(input, args=args, kwargs=kwargs), error_type=error_type, error_regex=error_regex)

def sample_inputs_scaled_mm(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_mat_e4m3 = partial(make_tensor, device=device, dtype=torch.float8_e4m3fn, requires_grad=requires_grad)
    make_mat_e5m2 = partial(make_tensor, device=device, dtype=torch.float8_e5m2, requires_grad=requires_grad)
    (M, N, K) = (15, 32, 16)
    samples = []
    mat1 = make_mat_e4m3((M, K))
    mat2 = make_mat_e4m3((K, N)).t().contiguous().t()
    samples.append(SampleInput(mat1, mat2))
    mat1 = make_mat_e4m3((M, K))
    mat2 = make_mat_e5m2((K, N)).t().contiguous().t()
    samples.append(SampleInput(mat1, mat2))
    mat1 = make_mat_e5m2((M, K))
    mat2 = make_mat_e4m3((K, N)).t().contiguous().t()
    samples.append(SampleInput(mat1, mat2))
    yield from samples

def sample_inputs_scaled_dot_product_attention(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    (batch, seq_q, seq_kv, num_heads, head_dim) = (4, 3, 6, 4, 8)
    dim_3_q_shape = (batch, seq_q, head_dim)
    dim_3_kv_shape = (batch, seq_kv, head_dim)
    dim_4_q_shape = (batch, num_heads, seq_q, head_dim)
    dim_4_kv_shape = (batch, num_heads, seq_kv, head_dim)
    broadcast_tuple = ((num_heads, seq_q, head_dim), (batch, num_heads, seq_kv, head_dim))
    qkv_shapes = [(dim_3_q_shape, dim_3_kv_shape), (dim_4_q_shape, dim_4_kv_shape), broadcast_tuple]
    samples = []
    for (qkv_shape, is_causal, dropout_p) in product(qkv_shapes, [True, False], [0.0, 0.5]):
        (shape_q, shape_kv) = qkv_shape
        samples.append(SampleInput(make(shape_q), make(shape_kv), make(shape_kv), is_causal=is_causal, dropout_p=dropout_p))
    diff_v_head_dim = SampleInput(make((batch, num_heads, seq_q, head_dim)), make((batch, num_heads, seq_kv, head_dim)), make((batch, num_heads, seq_kv, head_dim + 8)), is_causal=is_causal, dropout_p=dropout_p)
    samples.append(SampleInput(make((batch, num_heads, seq_q, head_dim)), make((batch, num_heads, seq_kv, head_dim)), make((batch, num_heads, seq_kv, head_dim)), attn_mask=make((seq_q, seq_kv)), is_causal=False, dropout_p=0.0))
    yield from samples

def sample_inputs_efficient_attention_forward(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    (batch, num_heads, head_dim) = (4, 4, 8)
    seq_q = 11
    seq_kv = 32
    dim_4_q_shape = (batch, num_heads, seq_q, head_dim)
    dim_4_kv_shape = (batch, num_heads, seq_kv, head_dim)
    qkv_shapes = [(dim_4_q_shape, dim_4_kv_shape)]
    samples = []
    mask_types = [1, 2]
    scales = [None, 1.0]
    for (qkv_shape, is_causal, dropout_p, mask_type, scale) in product(qkv_shapes, [True, False], [0.0, 0.5], mask_types, scales):
        (shape_q, shape_kv) = qkv_shape
        samples.append(SampleInput(make(shape_q).transpose(1, 2), make(shape_kv).transpose(1, 2), make(shape_kv).transpose(1, 2), bias=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, dropout_p=dropout_p, custom_mask_type=mask_type, compute_log_sumexp=requires_grad, scale=scale, causal_diagonal=None, seqlen_k=None))
    diff_v_head_dim = SampleInput(make((batch, seq_q, num_heads, head_dim)), make((batch, seq_kv, num_heads, head_dim)), make((batch, seq_kv, num_heads, head_dim + 8)), bias=None, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, dropout_p=dropout_p, custom_mask_type=0, compute_log_sumexp=requires_grad, scale=None, causal_diagonal=None, seqlen_k=None)
    samples.append(SampleInput(make((batch, seq_q, num_heads, head_dim)), make((batch, seq_kv, num_heads, head_dim)), make((batch, seq_kv, num_heads, head_dim)), bias=make(batch, num_heads, seq_q, seq_kv), cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=None, dropout_p=dropout_p, custom_mask_type=0, compute_log_sumexp=requires_grad, scale=None, causal_diagonal=None, seqlen_k=None))
    yield from samples

def sample_inputs_pairwise_distance(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    shape = (3,)
    batched_shape = (2, *shape)
    shapes_and_kwargs = [(shape, None), (batched_shape, None), (shape, dict(keepdim=True)), (batched_shape, dict(keepdim=True)), (shape, dict(p=5.0)), (shape, dict(p=-1.0)), (shape, dict(eps=1.0))]
    return (SampleInput(make(shape), args=(make(shape),), kwargs=kwargs) for (shape, kwargs) in shapes_and_kwargs)

def sample_inputs_pixel_shuffle(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield from (SampleInput(make_arg((1, 9, 2, 2)), upscale_factor=upscale_factor) for upscale_factor in (1, 3))
    yield from (SampleInput(make_arg(shape), upscale_factor=1) for shape in [(1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)])

def sample_inputs_pixel_unshuffle(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_arg = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield from (SampleInput(make_arg((1, 1, 6, 6)), downscale_factor=downscale_factor) for downscale_factor in (1, 3))
    yield from (SampleInput(make_arg(shape), downscale_factor=1) for shape in [(1, 0, 1, 1), (1, 1, 0, 1), (1, 1, 1, 0)])

def sample_inputs_binary_cross_entropy(op_info, device, dtype, requires_grad, logits=False, **kwargs):
    if False:
        return 10
    make = partial(make_tensor, device=device, dtype=dtype)
    make_prob = partial(make, low=1e-06, high=1)
    reductions = ('mean', 'sum', 'none')
    shapes_and_kwargs = [*[(shape, None) for shape in ((), (1,), (S,), (S, S), (S, S, S))], *[((S, S), dict(reduction=reduction)) for reduction in reductions], *[((S, S), dict(reduction=reduction, weight=make((S, S)))) for reduction in reductions]]
    if logits:
        shapes_and_kwargs.extend([((S, S), dict(reduction=reduction, pos_weight=make((S,), low=0))) for reduction in reductions])
    for (shape, kwargs) in shapes_and_kwargs:
        yield SampleInput((make if logits else make_prob)(shape, requires_grad=requires_grad), args=(make_prob(shape, requires_grad=requires_grad),), kwargs=kwargs)

def sample_inputs_allclose(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    sample_shapes = [(), S, (S, S, S)]
    atols = [0.01, 1e-16]
    rtols = [0.1, 0.5]
    eps = 1e-08
    for (s, rtol, atol) in product(sample_shapes, rtols, atols):
        t = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)
        close = (t + atol).detach().requires_grad_(requires_grad)
        yield SampleInput(t, close, rtol=rtol, atol=atol)
        a = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)
        b = make_tensor(s, device=device, dtype=dtype, requires_grad=requires_grad)
        yield SampleInput(a, b, rtol=rtol, atol=atol)

def sample_inputs_l1_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    yield from sample_inputs_loss(op_info, device, dtype, requires_grad, **kwargs)
    if dtype.is_complex:
        make = partial(make_tensor, (), device=device, requires_grad=requires_grad)
        yield SampleInput(make(dtype=dtype), args=(make(dtype=torch.double),))
        yield SampleInput(make(dtype=torch.double), args=(make(dtype=dtype),))

def error_inputs_l1_loss(op_info, device, **kwargs):
    if False:
        print('Hello World!')
    make = partial(make_tensor, device=device, dtype=torch.float32)
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5, 4),), kwargs={'reduction': 'abc'}), error_type=ValueError, error_regex='abc is not a valid value for reduction')
    yield ErrorInput(SampleInput(make(5, 4), args=(make(5),)), error_regex='(Attempting to broadcast a dimension of length|The size of tensor a \\(4\\) must match the size of tensor b \\(5\\) at non-singleton dimension 1)')

def sample_inputs_smooth_l1_loss(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    yield from sample_inputs_loss(op_info, device, dtype, requires_grad, **kwargs)
    make = partial(make_tensor, (S, S), device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make(low=0, high=2), args=(make(low=-2, high=0),), kwargs=dict(beta=5))
    yield SampleInput(make(), args=(make(),), kwargs=dict(beta=0))

def sample_inputs_kl_div(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_arg = partial(make_tensor, low=0.0, device=device, dtype=dtype, requires_grad=requires_grad)

    def make_log(shape):
        if False:
            for i in range(10):
                print('nop')
        out = torch.nn.functional.log_softmax(make_arg(shape), -1)
        out.requires_grad_(requires_grad)
        return out

    def make_prob(shape):
        if False:
            print('Hello World!')
        out = torch.nn.functional.softmax(make_arg(shape), -1)
        out.requires_grad_(requires_grad)
        return out
    shapes = ((2,), (2, 3))
    reductions = ('none', 'mean', 'batchmean', 'sum')
    for (shape, reduction, log_target) in product(shapes, reductions, (True, False)):
        input = make_log(shape)
        target = make_log(shape) if log_target else make_prob(shape)
        yield SampleInput(input, args=(target,), kwargs=dict(reduction=reduction, log_target=log_target))

def sample_inputs_pdist(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        while True:
            i = 10
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield from (SampleInput(make_input((n, m))) for (n, m) in itertools.product((1, S), repeat=2))
    yield from (SampleInput(make_input((S, S)), kwargs=dict(p=p)) for p in (0.0, 1.0, 2.0, 10.0, float('inf')))

def reference_pdist(input, p=2):
    if False:
        while True:
            i = 10
    pdist = scipy.spatial.distance.pdist
    if p == 0:
        output = pdist(input, 'hamming') * input.shape[1]
    elif p == float('inf'):
        output = pdist(input, lambda x, y: np.abs(x - y).max())
    else:
        output = pdist(input, 'minkowski', p=p)
    return output.astype(input.dtype)

def sample_inputs_diagflat(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    yield SampleInput(make_input(()))
    yield SampleInput(make_input((2,)))
    yield SampleInput(make_input((2, 2)))
    yield SampleInput(make_input((2,)), offset=1)
    yield SampleInput(make_input((2,)), offset=-1)

def sample_inputs_max_unpool(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    unpool_name_to_pool_method_dict = {'nn.functional.max_unpool1d': torch.nn.functional.max_pool1d, 'nn.functional.max_unpool2d': torch.nn.functional.max_pool2d, 'nn.functional.max_unpool3d': torch.nn.functional.max_pool3d}
    unpool_name_to_dim = {'nn.functional.max_unpool1d': 1, 'nn.functional.max_unpool2d': 2, 'nn.functional.max_unpool3d': 3}
    unpool_to_pool_name_dict = {k: f'nn.functional.{v.__name__}' for (k, v) in unpool_name_to_pool_method_dict.items()}
    pool_dim = unpool_name_to_dim[op_info.name]
    pool_method = unpool_name_to_pool_method_dict[op_info.name]
    pool_op_info = copy.copy(op_info)
    pool_op_info.name = unpool_to_pool_name_dict[op_info.name]
    for sample in sample_inputs_max_pool(pool_op_info, device, dtype, requires_grad, **kwargs):
        if sample.input.dim() != pool_dim + 2:
            continue
        if sample.kwargs['dilation'] != 1:
            continue
        if sample.kwargs['return_indices']:
            (pool, indices) = pool_method(sample.input, **sample.kwargs)
            arg = pool.detach().requires_grad_(requires_grad)
            sample_kwargs = {'kernel_size': sample.kwargs['kernel_size'], 'stride': sample.kwargs['stride'], 'padding': sample.kwargs['padding'], 'output_size': sample.input.size()}
            yield SampleInput(arg, args=(indices,), kwargs=sample_kwargs)

def sample_inputs_max_unpool_grad(op_info, device, dtype, requires_grad, **kwargs):
    if False:
        i = 10
        return i + 15
    for sample in sample_inputs_max_unpool(op_info, device, dtype, requires_grad, **kwargs):
        indices = sample.args[0]
        if indices.unique().numel() == indices.numel():
            yield sample

def sample_inputs_multi_head_attention_forward(opinfo, device, dtype, requires_grad, **kwargs):
    if False:
        print('Hello World!')
    make_input = partial(make_tensor, device=device, dtype=dtype, requires_grad=requires_grad)
    if requires_grad:
        bsz = 2
        is_batcheds = (True,)
        use_separate_proj_weights = (False,)
        emb_sizes = (2,)
        src_lens = (XS,)
        tgt_lens = (XS,)
        heads = (2,)
        dropouts = (0.5,)
        mask_types = ('2d',)
    else:
        bsz = 2
        is_batcheds = (False, True)
        use_separate_proj_weights = (False, True)
        emb_sizes = (2, 4)
        src_lens = (XS,)
        tgt_lens = (XS, S)
        heads = (1, 2)
        dropouts = (0.0, 0.5)
        mask_types = (None, '2d', '3d')
    for (is_batched, use_separate_proj_weight, mask_type, emb_size, src_len, tgt_len, num_heads, dropout_p) in itertools.product(is_batcheds, use_separate_proj_weights, mask_types, emb_sizes, src_lens, tgt_lens, heads, dropouts):
        attn_mask = None
        if mask_type == '2d':
            attn_mask = make_input(src_len, tgt_len)
        elif mask_type == '3d':
            attn_mask = make_input((bsz if is_batched else 1) * num_heads, src_len, tgt_len)
        if is_batched:
            q = make_input(src_len, bsz, emb_size)
            k = make_input(tgt_len, bsz, emb_size)
            v = make_input(tgt_len, bsz, emb_size)
        else:
            q = make_input(src_len, emb_size)
            k = make_input(tgt_len, emb_size)
            v = make_input(tgt_len, emb_size)
        if use_separate_proj_weight:
            in_proj_weight = None
            q_proj_weight = make_input(emb_size, emb_size)
            k_proj_weight = make_input(emb_size, emb_size)
            v_proj_weight = make_input(emb_size, emb_size)
        else:
            in_proj_weight = make_input(emb_size * 3, emb_size)
            q_proj_weight = None
            k_proj_weight = None
            v_proj_weight = None
        bias_k = make_input(emb_size)
        bias_v = make_input(emb_size)
        in_proj_bias = make_input(emb_size * 3)
        out_proj_weight = make_input(emb_size, emb_size)
        out_proj_bias = make_input(emb_size)
        sample_args = (k, v, emb_size, num_heads, in_proj_weight, in_proj_bias, bias_k, bias_v, False, dropout_p, out_proj_weight, out_proj_bias)
        sample_kwargs = {'q_proj_weight': q_proj_weight, 'k_proj_weight': k_proj_weight, 'v_proj_weight': v_proj_weight, 'attn_mask': attn_mask, 'training': True if dropout_p > 0.0 else False, 'use_separate_proj_weight': use_separate_proj_weight}
        yield SampleInput(q, args=sample_args, kwargs=sample_kwargs)
NUM_SIZE0_TENSORS = 10000
foreach_num_tensors = [20, 23] if not TEST_WITH_SLOW else [23, 30, 300]
_foreach_inputs_default_kwargs = {'noncontiguous': False, 'same_size': False, 'low': None, 'high': None}

class ForeachRightmostArgType(enum.Enum):
    TensorList = enum.auto()
    ScalarList = enum.auto()
    Scalar = enum.auto()
    Tensor = enum.auto()

class ForeachSampleInput(SampleInput):
    ref_args: Any
    disable_fastpath: bool

    def __init__(self, *args, disable_fastpath=False, ref_args=None, **kwargs):
        if False:
            return 10
        super().__init__(*args, **kwargs)
        self.ref_args = ref_args or self.args
        self.disable_fastpath = disable_fastpath

class foreach_inputs_sample_func:

    def __init__(self, arity: int, rightmost_supports_scalar: bool, rightmost_supports_scalarlist: bool, rightmost_supports_tensor: bool=False) -> None:
        if False:
            return 10
        self.arity = arity
        self._set_rightmost_arg_types(rightmost_supports_scalar, rightmost_supports_scalarlist, rightmost_supports_tensor)

    def _set_rightmost_arg_types(self, rightmost_supports_scalar: bool, rightmost_supports_scalarlist: bool, rightmost_supports_tensor: bool) -> None:
        if False:
            while True:
                i = 10
        self._rightmost_arg_types = [ForeachRightmostArgType.TensorList]
        if self.arity > 1:
            if rightmost_supports_scalar:
                self._rightmost_arg_types.append(ForeachRightmostArgType.Scalar)
            if rightmost_supports_scalarlist:
                self._rightmost_arg_types.append(ForeachRightmostArgType.ScalarList)
            if rightmost_supports_tensor:
                self._rightmost_arg_types.append(ForeachRightmostArgType.Tensor)

    def _sample_rightmost_arg(self, opinfo, rightmost_arg_type, device, dtype, num_tensors, **_foreach_inputs_kwargs):
        if False:
            print('Hello World!')
        if rightmost_arg_type == ForeachRightmostArgType.TensorList:
            return [sample_inputs_foreach(None, device, dtype, num_tensors, **_foreach_inputs_kwargs)]
        if rightmost_arg_type == ForeachRightmostArgType.Tensor:
            return [make_tensor((), device=device, dtype=dtype, noncontiguous=_foreach_inputs_kwargs['noncontiguous'], requires_grad=_foreach_inputs_kwargs.get('requires_grad', False))]
        should_use_simpler_scalars = opinfo.name == '_foreach_pow' and dtype in (torch.float16, torch.bfloat16)

        def sample_float():
            if False:
                print('Hello World!')
            s = random.random()
            if should_use_simpler_scalars:
                return 1.0 if s > 0.5 else 2.0
            else:
                return 1.0 - s
        high = 2 if should_use_simpler_scalars else 9
        if rightmost_arg_type == ForeachRightmostArgType.ScalarList:
            return [[random.randint(0, high) + 1 for _ in range(num_tensors)], [sample_float() for _ in range(num_tensors)], [complex(sample_float(), sample_float()) for _ in range(num_tensors)], [True for _ in range(num_tensors)], [1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(num_tensors - 3)], [True, 1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(num_tensors - 4)]]
        if rightmost_arg_type == ForeachRightmostArgType.Scalar:
            return (random.randint(1, high + 1), sample_float(), True, complex(sample_float(), sample_float()))
        raise AssertionError(f'Invalid rightmost_arg_type of {rightmost_arg_type}')

    def _should_disable_fastpath(self, opinfo, rightmost_arg, rightmost_arg_type, dtype):
        if False:
            i = 10
            return i + 15
        if self.arity == 1:
            if 'foreach_abs' in opinfo.name and dtype in complex_types():
                return True
            if opinfo.ref in (torch.abs, torch.neg):
                return False
            return dtype in integral_types_and(torch.bool)
        if self.arity < 2 or rightmost_arg_type == ForeachRightmostArgType.Tensor:
            return None
        if 'foreach_pow' in opinfo.name and dtype in integral_types():
            return True
        if rightmost_arg_type == ForeachRightmostArgType.TensorList:
            disable_fastpath = 'foreach_div' in opinfo.name and dtype in integral_types_and(torch.bool)
            if 'foreach_add' in opinfo.name and dtype == torch.bool:
                disable_fastpath = True
            return disable_fastpath
        elif rightmost_arg_type == ForeachRightmostArgType.Scalar:
            disable_fastpath = 'foreach_div' in opinfo.name and dtype in integral_types_and(torch.bool)
            if isinstance(rightmost_arg, bool):
                disable_fastpath |= dtype == torch.bool
                if opinfo.ref in (torch.add, torch.mul):
                    disable_fastpath = False
            elif isinstance(rightmost_arg, int):
                disable_fastpath |= dtype == torch.bool
            elif isinstance(rightmost_arg, float):
                disable_fastpath |= dtype in integral_types_and(torch.bool)
            elif isinstance(rightmost_arg, complex):
                disable_fastpath |= dtype not in complex_types()
            else:
                raise AssertionError(f'Invalid scalar of type {rightmost_arg_type} - {rightmost_arg}')
            return disable_fastpath
        elif rightmost_arg_type == ForeachRightmostArgType.ScalarList:
            disable_fastpath = opinfo.ref == torch.div and dtype in integral_types_and(torch.bool)
            elmt_t = type(rightmost_arg[0])
            has_same_type = all((isinstance(v, elmt_t) for v in rightmost_arg))
            if not has_same_type:
                return dtype not in complex_types()
            if isinstance(rightmost_arg[0], bool):
                if ('foreach_add' in opinfo.name or 'foreach_mul' in opinfo.name) and dtype == torch.bool:
                    disable_fastpath = False
            elif isinstance(rightmost_arg[0], int):
                disable_fastpath |= dtype == torch.bool
            elif isinstance(rightmost_arg[0], float):
                disable_fastpath |= dtype in integral_types_and(torch.bool)
            elif isinstance(rightmost_arg[0], complex):
                disable_fastpath |= dtype not in complex_types()
            else:
                raise AssertionError(f'Invalid scalarlist of {rightmost_arg}')
            return disable_fastpath
        else:
            raise AssertionError(f'Invalid rightmost_arg_type of {rightmost_arg_type}')

    def _sample_kwargs(self, opinfo, rightmost_arg, rightmost_arg_type, dtype):
        if False:
            i = 10
            return i + 15
        kwargs = {}
        if rightmost_arg_type == ForeachRightmostArgType.TensorList and opinfo.supports_alpha_param:
            if dtype in integral_types_and(torch.bool):
                kwargs['alpha'] = 3
            elif dtype.is_complex:
                kwargs['alpha'] = complex(3, 3)
            else:
                kwargs['alpha'] = 3.14
        if self.arity > 1:
            kwargs['disable_fastpath'] = self._should_disable_fastpath(opinfo, rightmost_arg, rightmost_arg_type, dtype)
        return kwargs

    def sample_zero_size_tensor_inputs(self, opinfo, device, dtype, requires_grad, **kwargs):
        if False:
            return 10
        assert 'num_input_tensors' not in kwargs
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for (k, v) in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for rightmost_arg_type in self._rightmost_arg_types:
            zero_size_foreach_inputs_kwargs = copy.deepcopy(_foreach_inputs_kwargs)
            zero_size_foreach_inputs_kwargs['zero_size'] = True
            input = sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, **zero_size_foreach_inputs_kwargs)
            if self.arity > 1:
                args = [sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, **zero_size_foreach_inputs_kwargs) for _ in range(self.arity - 2)]
                args.append(self._sample_rightmost_arg(opinfo, ForeachRightmostArgType.TensorList, device, dtype, NUM_SIZE0_TENSORS, **zero_size_foreach_inputs_kwargs)[0])
                kwargs = self._sample_kwargs(opinfo, args[-1], ForeachRightmostArgType.TensorList, dtype, zero_size=True)
            else:
                args = []
                kwargs = {}
                if opinfo.ref in (torch.abs, torch.neg):
                    kwargs['disable_fastpath'] = False
                else:
                    kwargs['disable_fastpath'] = dtype in integral_types_and(torch.bool)
            yield ForeachSampleInput(input, *args, **kwargs)

    def __call__(self, opinfo, device, dtype, requires_grad, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        num_input_tensors_specified = 'num_input_tensors' in kwargs
        num_input_tensors = kwargs.pop('num_input_tensors') if num_input_tensors_specified else foreach_num_tensors
        assert isinstance(num_input_tensors, list)
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for (k, v) in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        _foreach_inputs_kwargs['zero_size'] = False
        for (num_tensors, rightmost_arg_type, intersperse_empty_tensors) in itertools.product(num_input_tensors, self._rightmost_arg_types, (True, False)):
            if intersperse_empty_tensors and (num_tensors != max(num_input_tensors) or str(device) == 'cpu'):
                continue
            _foreach_inputs_kwargs['intersperse_empty_tensors'] = intersperse_empty_tensors
            input = sample_inputs_foreach(None, device, dtype, num_tensors, **_foreach_inputs_kwargs)
            args = []
            if self.arity > 1:
                args = [sample_inputs_foreach(None, device, dtype, num_tensors, **_foreach_inputs_kwargs) for _ in range(self.arity - 2)]
                rightmost_arg_list = self._sample_rightmost_arg(opinfo, rightmost_arg_type, device, dtype, num_tensors, **_foreach_inputs_kwargs)
                for rightmost_arg in rightmost_arg_list:
                    args.append(rightmost_arg)
                    kwargs = self._sample_kwargs(opinfo, rightmost_arg, rightmost_arg_type, dtype)
                    ref_args = args
                    if rightmost_arg_type in (ForeachRightmostArgType.Scalar, ForeachRightmostArgType.Tensor):
                        ref_args = args[:-1] + [[args[-1] for _ in range(num_tensors)]]
                    sample = ForeachSampleInput(input, *args, ref_args=ref_args, **kwargs)
                    yield sample
                    args.pop()
            else:
                yield ForeachSampleInput(input, *args, disable_fastpath=self._should_disable_fastpath(opinfo, None, None, dtype))

class foreach_norm_sample_func(foreach_inputs_sample_func):

    def sample_zero_size_tensor_inputs(self, opinfo, device, dtype, requires_grad, **kwargs):
        if False:
            print('Hello World!')
        assert 'num_input_tensors' not in kwargs
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for (k, v) in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for ord in (0, 1, 2, -1, -2):
            input = sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, zero_size=True, **_foreach_inputs_kwargs)
            disable_fastpath = True
            if ord in (1, 2) and dtype in floating_types_and(torch.half, torch.bfloat16):
                disable_fastpath = False
            yield ForeachSampleInput(input, **{'ord': ord, 'disable_fastpath': disable_fastpath})

    def __call__(self, opinfo, device, dtype, requires_grad, **kwargs):
        if False:
            while True:
                i = 10
        num_input_tensors = kwargs.pop('num_input_tensors', foreach_num_tensors)
        assert isinstance(num_input_tensors, list)
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for (k, v) in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for (num_tensors, ord) in product(num_input_tensors, (0, 1, 2, -1, -2)):
            input = sample_inputs_foreach(None, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs)
            disable_fastpath = True
            if ord in (1, 2) and dtype in floating_types_and(torch.half, torch.bfloat16):
                disable_fastpath = False
            yield ForeachSampleInput(input, **{'ord': ord, 'disable_fastpath': disable_fastpath})

class foreach_lerp_sample_func(foreach_inputs_sample_func):

    def _sample_rightmost_arg(self, opinfo, rightmost_arg_type, device, dtype, num_tensors, **_foreach_inputs_kwargs):
        if False:
            i = 10
            return i + 15
        if rightmost_arg_type == ForeachRightmostArgType.TensorList:
            return [sample_inputs_foreach(None, device, dtype, num_tensors, **_foreach_inputs_kwargs)]
        if rightmost_arg_type == ForeachRightmostArgType.ScalarList:
            return [[random.randint(0, 9) + 1 for _ in range(num_tensors)], [1.0 - random.random() for _ in range(num_tensors)], [complex(1.0 - random.random(), 1.0 - random.random()) for _ in range(num_tensors)], [True for _ in range(num_tensors)], [1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(num_tensors - 3)], [True, 1, 2.0, 3.0 + 4.5j] + [3.0 for _ in range(num_tensors - 4)]]
        if rightmost_arg_type == ForeachRightmostArgType.Scalar:
            return [random.random()]
        raise AssertionError(f'Invalid rightmost_arg_type of {rightmost_arg_type}')

class foreach_pointwise_sample_func(foreach_inputs_sample_func):

    def __init__(self, arity: int=3, rightmost_supports_scalar: bool=False, rightmost_supports_scalarlist: bool=False):
        if False:
            while True:
                i = 10
        super().__init__(arity, rightmost_supports_scalar, rightmost_supports_scalarlist)

    def _should_disable_fastpath(self, opinfo, rightmost_arg, rightmost_arg_type, dtype):
        if False:
            for i in range(10):
                print('nop')
        return dtype in integral_types_and(torch.bool) and opinfo.ref in (torch.addcmul,)

    def sample_zero_size_tensor_inputs(self, opinfo, device, dtype, requires_grad, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        assert 'num_input_tensors' not in kwargs
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for (k, v) in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        input = sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, zero_size=True, **_foreach_inputs_kwargs)
        args = [sample_inputs_foreach(None, device, dtype, NUM_SIZE0_TENSORS, zero_size=True, **_foreach_inputs_kwargs) for _ in range(2)]
        kwargs['values'] = None
        kwargs.update(self._sample_kwargs(opinfo, args[-1], ForeachRightmostArgType.TensorList, dtype))
        yield ForeachSampleInput(input, *args, **kwargs)

    def __call__(self, opinfo, device, dtype, requires_grad, **kwargs):
        if False:
            print('Hello World!')
        num_input_tensors_specified = 'num_input_tensors' in kwargs
        num_input_tensors = kwargs.pop('num_input_tensors') if num_input_tensors_specified else foreach_num_tensors
        assert isinstance(num_input_tensors, list)
        _foreach_inputs_kwargs = {k: kwargs.pop(k, v) for (k, v) in _foreach_inputs_default_kwargs.items()}
        _foreach_inputs_kwargs['requires_grad'] = requires_grad
        for (num_tensors, rightmost_arg_type) in itertools.product(num_input_tensors, self._rightmost_arg_types):
            input = sample_inputs_foreach(None, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs)
            args = [sample_inputs_foreach(None, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs) for _ in range(2 - int(rightmost_arg_type == ForeachRightmostArgType.TensorList))]
            rightmost_arg_list = self._sample_rightmost_arg(opinfo, rightmost_arg_type, device, dtype, num_tensors, zero_size=False, **_foreach_inputs_kwargs)
            for rightmost_arg in rightmost_arg_list:
                kwargs = {}
                if rightmost_arg_type == ForeachRightmostArgType.TensorList:
                    args.append(rightmost_arg)
                else:
                    kwargs['values'] = rightmost_arg
                kwargs.update(self._sample_kwargs(opinfo, rightmost_arg, rightmost_arg_type, dtype))
                assert len(args) == 2, f'len(args)={len(args)!r}'
                sample = ForeachSampleInput(input, *args, **kwargs)
                yield sample
                if rightmost_arg_type == ForeachRightmostArgType.TensorList:
                    args.pop()
foreach_unary_op_db: List[OpInfo] = [ForeachFuncInfo('exp', foreach_inputs_sample_func(1, False, False), backward_requires_result=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('acos', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('asin', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('atan', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('cos', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('cosh', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('log', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('log10', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('log2', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('tan', foreach_inputs_sample_func(1, False, False), backward_requires_result=True, decorators=(DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0003, rtol=2e-05)}), 'TestForeach', 'test_parity', device_type='cuda'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('tanh', foreach_inputs_sample_func(1, False, False), backward_requires_result=True, decorators=(DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.005, rtol=0.0001)}), 'TestForeach', 'test_parity', device_type='cuda'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('sin', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('sinh', foreach_inputs_sample_func(1, False, False), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('neg', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and_complex(), dtypesIfCUDA=all_types_and_complex()), ForeachFuncInfo('sqrt', foreach_inputs_sample_func(1, False, False), dtypes=floating_and_complex_types_and(torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half), backward_requires_result=True), ForeachFuncInfo('ceil', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('erf', foreach_inputs_sample_func(1, False, False), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('erfc', foreach_inputs_sample_func(1, False, False), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('expm1', foreach_inputs_sample_func(1, False, False), dtypes=floating_and_complex_types_and(torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), backward_requires_result=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('floor', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('log1p', foreach_inputs_sample_func(1, False, False), dtypes=floating_and_complex_types_and(torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('round', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('frac', foreach_inputs_sample_func(1, False, False), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('reciprocal', foreach_inputs_sample_func(1, False, False), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half), backward_requires_result=True), ForeachFuncInfo('sigmoid', foreach_inputs_sample_func(1, False, False), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half), backward_requires_result=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('trunc', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('abs', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and_complex_and(torch.bfloat16, torch.half), dtypesIfCUDA=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool), supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('In-place abs not supported for complex tensors'), 'TestMeta', 'test_dispatch_meta_inplace', dtypes=complex_types()), DecorateInfo(unittest.skip('In-place abs not supported for complex tensors'), 'TestMeta', 'test_meta_inplace', dtypes=complex_types()), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace', dtypes=complex_types()), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace', dtypes=complex_types()), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace', dtypes=complex_types()), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace', dtypes=complex_types()), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides', dtypes=complex_types()), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides', dtypes=complex_types()))), ForeachFuncInfo('zero', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and_complex_and(torch.bfloat16, torch.half), has_no_out_of_place=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('sign', foreach_inputs_sample_func(1, False, False), dtypes=floating_types_and(torch.bool, torch.bfloat16, torch.half), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16)), ForeachFuncInfo('lgamma', foreach_inputs_sample_func(1, False, False), dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half), dtypesIfCUDA=all_types_and(torch.bool, torch.float16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides')))]
foreach_binary_op_db: List[OpInfo] = [ForeachFuncInfo('add', foreach_inputs_sample_func(2, True, True, True), dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), supports_alpha_param=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('sub', foreach_inputs_sample_func(2, True, True), dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), supports_alpha_param=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('mul', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), sample_inputs_func=foreach_inputs_sample_func(2, True, True, True), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('div', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), sample_inputs_func=foreach_inputs_sample_func(2, True, True, True), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('clamp_min', foreach_inputs_sample_func(2, True, True), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('clamp_max', foreach_inputs_sample_func(2, True, True), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('minimum', foreach_inputs_sample_func(2, True, True), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16), supports_forward_ad=False, supports_inplace_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('maximum', foreach_inputs_sample_func(2, True, True), dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16), supports_forward_ad=False, supports_inplace_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('pow', dtypes=all_types_and(torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16), supports_alpha_param=False, supports_scalar_self_arg=True, sample_inputs_func=foreach_inputs_sample_func(2, True, True), supports_autograd=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides')), supports_forward_ad=True, backward_requires_result=True), ForeachFuncInfo('copy', foreach_inputs_sample_func(2, False, False), dtypes=all_types_and_complex_and(torch.bfloat16, torch.half), has_no_out_of_place=True, supports_forward_ad=False, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides')))]
foreach_pointwise_op_db: List[ForeachFuncInfo] = [ForeachFuncInfo('addcmul', foreach_pointwise_sample_func(4, True, True), dtypes=all_types_and_complex(), dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), ForeachFuncInfo('addcdiv', sample_inputs_func=foreach_pointwise_sample_func(4, True, True), dtypes=all_types_and_complex(), dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides')))]
foreach_reduce_op_db: List[ForeachFuncInfo] = [ForeachFuncInfo('norm', foreach_norm_sample_func(1, False, False), dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides')))]
foreach_other_op_db: List[ForeachFuncInfo] = [ForeachFuncInfo('lerp', foreach_lerp_sample_func(3, True, False), dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides')))]

def reference_sign(x):
    if False:
        return 10
    if x.dtype == np.bool_:
        return np.sign(x, dtype=np.uint8).astype(np.bool_)
    return np.sign(x)

def reference_sgn(x):
    if False:
        print('Hello World!')
    if x.dtype not in [np.complex64, np.complex128]:
        return reference_sign(x)
    out = x / np.abs(x)
    if out.ndim == 0:
        if x == 0:
            return np.array(complex(0, 0), dtype=x.dtype)
        return out
    mask = x == 0
    out[mask] = complex(0, 0)
    return out

def reference_sigmoid(x):
    if False:
        return 10
    if x.dtype in [np.complex64, np.complex128]:
        return 1 / (1 + np.exp(-x))
    return scipy.special.expit(x)

def reference_logsigmoid(x):
    if False:
        for i in range(10):
            print('nop')
    return np.where(x < 0, x - np.log1p(np.exp(x)), -np.log1p(np.exp(-x)))

def reference_hardsigmoid(x):
    if False:
        while True:
            i = 10
    intermediate = x / 6 + 0.5
    y = np.clip(intermediate, 0, None)
    return np.where(y > 1, 1, y).astype(x.dtype)

def reference_lgamma(x):
    if False:
        i = 10
        return i + 15
    if x.dtype.kind == 'f':
        x = np.where(x == float('-inf'), np.array(float('inf'), dtype=x.dtype), x)
    out = scipy.special.gammaln(x)
    if x.dtype == np.float16:
        out = out.astype(np.float16)
    return out

def reference_mvlgamma(x, d):
    if False:
        for i in range(10):
            print('nop')
    if x.dtype == np.float16:
        return scipy.special.multigammaln(x, d).astype(np.float16)
    return scipy.special.multigammaln(x, d)

def reference_softplus(input, beta=1, threshold=20):
    if False:
        while True:
            i = 10
    non_linear = input * beta <= threshold
    output = input.copy()
    output[non_linear] = np.log(1 + np.exp(beta * input[non_linear])) / beta
    return output

def reference_gelu(X, *, approximate='none'):
    if False:
        while True:
            i = 10

    def _gelu_ref(X):
        if False:
            return 10
        return X * stats.norm.cdf(X)

    def _tanh_gelu_ref(X):
        if False:
            for i in range(10):
                print('nop')
        M_SQRT_2_PI = math.sqrt(2 / math.pi)
        Z = M_SQRT_2_PI * (X + 0.044715 * np.power(X, 3.0))
        return 0.5 * X * (1.0 + np.tanh(Z))
    if approximate == 'tanh':
        return _tanh_gelu_ref(X)
    else:
        return _gelu_ref(X)

def reference_one_hot(a: np.ndarray, num_classes: int=-1) -> np.ndarray:
    if False:
        for i in range(10):
            print('nop')
    if num_classes == -1:
        num_classes = int(np.amax(a) + 1)
    idcs = a.reshape(-1) + np.arange(0, a.size, dtype=np.int64) * num_classes
    one_hot = np.zeros((a.size, num_classes), dtype=a.dtype)
    np.put(one_hot, idcs, 1)
    return one_hot.reshape(*a.shape, -1)

def reference_mse_loss(input, target, reduction='mean'):
    if False:
        print('Hello World!')
    se = (input - target) ** 2
    if reduction == 'mean':
        return np.mean(se)
    elif reduction == 'sum':
        return np.sum(se)
    else:
        return se

def wrapper_set_seed(op, *args, **kwargs):
    if False:
        return 10
    'Wrapper to set seed manually for some functions like dropout\n    See: https://github.com/pytorch/pytorch/pull/62315#issuecomment-896143189 for more details.\n    '
    with freeze_rng_state():
        torch.manual_seed(42)
        return op(*args, **kwargs)

def reference_layer_norm(inp: np.ndarray, normalized_shape: Tuple[int], weight=None, bias=None, eps=1e-05):
    if False:
        return 10
    return reference_native_layer_norm(inp, normalized_shape, weight, bias, eps)[0]

def reference_native_layer_norm(inp: np.ndarray, normalized_shape: Tuple[int], weight, bias, eps):
    if False:
        while True:
            i = 10
    feature_size = np.prod(normalized_shape)
    inp_view = inp.reshape(-1, feature_size)
    mean = inp_view.mean(axis=-1, keepdims=True)
    var = inp_view.var(axis=-1, ddof=0, keepdims=True)
    Y = (inp_view - mean) / np.sqrt(var + eps)
    if weight is None and bias is not None:
        Y = Y + bias.reshape(-1)
    elif weight is not None and bias is None:
        Y = Y * weight.reshape(-1)
    elif weight is not None and bias is not None:
        Y = Y * weight.reshape(-1) + bias.reshape(-1)
    axis = inp.ndim - len(normalized_shape)
    stat_shape = inp.shape[:axis] + (1,) * len(normalized_shape)
    return (Y.reshape(*inp.shape), mean.reshape(stat_shape), (1.0 / np.sqrt(var + eps)).reshape(stat_shape))

def reference_group_norm(inp: np.ndarray, num_groups: int, weight=None, bias=None, eps=1e-05):
    if False:
        return 10
    inp_view = inp
    if np.prod(inp.shape) != 0:
        inp_view = inp.reshape((inp.shape[0], num_groups, -1))
    mean = inp_view.mean(axis=-1, keepdims=True)
    var = inp_view.var(axis=-1, ddof=0, keepdims=True)
    Y = (inp_view - mean) / np.sqrt(var + eps)
    Y = Y.reshape(inp.shape)
    if weight is not None:
        if len(Y.shape) > 2:
            weight = np.expand_dims(weight, [0] + [idx + 2 for idx in range(inp.ndim - 2)])
        Y = Y * weight
    if bias is not None:
        if len(Y.shape) > 2:
            bias = np.expand_dims(bias, [0] + [idx + 2 for idx in range(inp.ndim - 2)])
        Y = Y + bias
    return Y

def reference_searchsorted(sorted_sequence, boundary, out_int32=False, right=False, side='left', sorter=None):
    if False:
        return 10
    side = 'right' if right or side == 'right' else 'left'
    if len(sorted_sequence.shape) == 1:
        ret = np.searchsorted(sorted_sequence, boundary, side=side, sorter=sorter)
        return ret.astype(np.int32) if out_int32 else ret
    elif sorted_sequence.shape[0] == 0:
        if sorter is not None:
            sorter = sorter.flatten()
        ret = np.searchsorted(sorted_sequence.flatten(), boundary.flatten(), side=side, sorter=sorter)
        ret = ret.astype(np.int32) if out_int32 else ret
        return ret.reshape(boundary.shape)
    else:
        orig_shape = boundary.shape
        num_splits = np.prod(sorted_sequence.shape[:-1])
        splits = range(0, num_splits)
        (sorted_sequence, boundary) = (sorted_sequence.reshape(num_splits, -1), boundary.reshape(num_splits, -1))
        if sorter is not None:
            sorter = sorter.reshape(num_splits, -1)
        split_sequence = [sorted_sequence[i] for i in splits]
        split_boundary = [boundary[i] for i in splits]
        split_sorter = [sorter[i] if sorter is not None else None for i in splits]
        split_ret = [np.searchsorted(s_seq, b, side=side, sorter=s_sort) for (s_seq, b, s_sort) in zip(split_sequence, split_boundary, split_sorter)]
        split_ret = [i.astype(np.int32) for i in split_ret] if out_int32 else split_ret
        return np.stack(split_ret).reshape(orig_shape)

def loss_reference_reduction_wrapper(fn):
    if False:
        while True:
            i = 10

    def wrapper(input, target, *, size_average=None, reduce=None, reduction='mean', **other_kwargs):
        if False:
            for i in range(10):
                print('nop')
        if size_average is not None or reduce is not None:
            raise RuntimeError("The keyword arguments 'size_average' and 'reduce' are deprecated and not supported by this wrapper")
        output = fn(input, target, **other_kwargs)
        if reduction == 'mean':
            return np.mean(output)
        elif reduction == 'sum':
            return np.sum(output)
        else:
            return output
    return wrapper

@loss_reference_reduction_wrapper
def reference_smooth_l1_loss(input, target, beta=1.0):
    if False:
        for i in range(10):
            print('nop')
    diff = input - target
    abs_diff = np.abs(diff)
    above_threshold = abs_diff >= beta
    loss = np.empty_like(input)
    loss[above_threshold] = abs_diff[above_threshold] - 0.5 * beta
    loss[~above_threshold] = diff[~above_threshold] ** 2 / (2 * beta)
    return loss

def reference_std_var(f):
    if False:
        for i in range(10):
            print('nop')
    "Forwards unbiased/correction kwargs as NumPy's equivalent ddof"
    g = reference_reduction_numpy(f)

    @wraps(g)
    def wrapper(x: np.ndarray, *args, **kwargs):
        if False:
            print('Hello World!')
        assert not ('unbiased' in kwargs and 'correction' in kwargs)
        if 'unbiased' in kwargs:
            kwargs['ddof'] = int(kwargs.pop('unbiased'))
        elif 'correction' in kwargs:
            kwargs['ddof'] = kwargs.pop('correction')
        return g(x, *args, **kwargs)
    return wrapper

def generate_std_var_kwargs(t: torch.Tensor, **kwargs):
    if False:
        i = 10
        return i + 15
    'Generates unbiased/correction kwargs for std/var operators'
    yield ((), {'unbiased': True})
    yield ((), {'unbiased': False})
    if 'dim' in kwargs and 'keepdim' in kwargs:
        yield ((), {'correction': 0})
        yield ((), {'correction': 1})
        numel = torch.tensor(t.shape)[kwargs.get('dim')].prod()
        yield ((), {'correction': numel // 2})

def error_inputs_mean(op_info, device, is_ref=False, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    if is_ref:
        err_msg1 = 'mean\\(\\): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: torch.int64'
    else:
        err_msg1 = 'mean\\(\\): could not infer output dtype. Input dtype must be either a floating point or complex dtype. Got: Long'
    yield ErrorInput(SampleInput(make_tensor((3, 4, 5), dtype=torch.int64, device=device), []), error_regex=err_msg1)
    if is_ref:
        err_msg2 = 'mean\\(\\): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: torch.int64'
    else:
        err_msg2 = 'mean\\(\\): could not infer output dtype. Optional dtype must be either a floating point or complex dtype. Got: Long'
    yield ErrorInput(SampleInput(make_tensor((3, 4, 5), dtype=torch.float32, device=device), [], dtype=torch.int64), error_regex=err_msg2)
    if is_ref:
        err_msg3 = 'Expected out tensor to have dtype torch.float64, but got torch.float32 instead'
    else:
        err_msg3 = 'Expected out tensor to have dtype double, but got float instead'
    yield ErrorInput(SampleInput(make_tensor((3, 4, 5), dtype=torch.int64, device=device), [], dtype=torch.float64, out=make_tensor([], dtype=torch.float32, device=device)), error_regex=err_msg3)

def reference_flatten(input, start_dim=0, end_dim=-1):
    if False:
        while True:
            i = 10
    in_shape = input.shape
    in_rank = len(in_shape)
    for d in (start_dim, end_dim):
        if not (in_rank == 0 and d in (-1, 0) or -in_rank <= d < in_rank):
            raise IndexError(f'Dimension out of range (expected to be in range of [{-in_rank}, {in_rank - 1}], but got {d}')
    end_dim = end_dim if end_dim >= 0 else in_rank + end_dim
    start_dim = start_dim if start_dim >= 0 else in_rank + start_dim
    if in_rank == 0:
        end_dim = start_dim
    if end_dim < start_dim:
        raise RuntimeError('flatten() has invalid args: start_dim cannot come after end_dim')
    flatten_bit_dim = functools.reduce(operator.mul, in_shape[start_dim:end_dim + 1], 1)
    out_shape = in_shape[:start_dim] + (flatten_bit_dim,) + in_shape[end_dim + 1:]
    return np.reshape(input, out_shape)
op_db: List[OpInfo] = [UnaryUfuncInfo('abs', aliases=('absolute',), ref=np.abs, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.chalf), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), skips=(DecorateInfo(unittest.skip('In-place abs not supported for complex tensors'), 'TestBwdGradients', 'test_inplace_grad', dtypes=(torch.cdouble,)), DecorateInfo(unittest.skip('In-place abs not supported for complex tensors'), 'TestBwdGradients', 'test_inplace_gradgrad', dtypes=(torch.cdouble,)), DecorateInfo(unittest.skip('In-place abs not supported for complex tensors'), 'TestFwdGradients', 'test_inplace_forward_mode_AD', dtypes=(torch.cdouble,)), DecorateInfo(unittest.skip('In-place abs not supported for complex tensors'), 'TestSparseUnaryUfuncs', 'test_inplace', dtypes=(torch.cdouble, torch.cfloat, torch.chalf)), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=[torch.int8], active_if=TEST_WITH_ASAN), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_out_arg_all_dtypes', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace', dtypes=(torch.cdouble, torch.cfloat, torch.chalf)), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace', dtypes=(torch.cdouble, torch.cfloat, torch.chalf)), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace', dtypes=(torch.cdouble, torch.cfloat, torch.chalf)), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides', dtypes=(torch.cdouble, torch.cfloat, torch.chalf))), supports_fwgrad_bwgrad=True, assert_autodiffed=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_forward_ad=True), UnaryUfuncInfo('acos', aliases=('arccos',), ref=np.arccos, domain=(-1, 1), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.1, torch.complex64: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_grad', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_method_grad', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_inplace_grad', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_inplace_forward_mode_AD', dtypes=[torch.cdouble], active_if=IS_WINDOWS))), UnaryUfuncInfo('acosh', aliases=('arccosh',), ref=np.arccosh, domain=(1, None), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), decorators=(precisionOverride({torch.bfloat16: 0.05}),), supports_inplace_autograd=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS)), reference_numerics_filter=NumericsFilter(condition=lambda x: x < 1 if not x.is_complex() else torch.zeros_like(x, dtype=torch.bool), safe_val=2)), BinaryUfuncInfo('add', ref=lambda input, other, *, alpha=1: np.add(input, other) if alpha == 1 else np.add(input, np.multiply(alpha, other)), dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), assert_autodiffed=True, sample_inputs_func=sample_inputs_add_sub, supports_fwgrad_bwgrad=True, supports_forward_ad=True, supports_two_python_scalars=True, decorators=(DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.01, rtol=0)}), 'TestBinaryUfuncs', 'test_reference_numerics'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bool,)), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_numpy_refs', dtypes=(torch.complex128,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=(torch.complex64, torch.complex128)))), OpInfo('item', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.item, inp, *args, **kwargs), ref=np.ndarray.item, method_variant=None, dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.chalf, torch.bool), supports_out=False, supports_autograd=False, error_inputs_func=error_inputs_item, sample_inputs_func=sample_inputs_item, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestFakeTensor', 'test_fake_autocast'), DecorateInfo(unittest.expectedFailure, 'TestFakeTensor', 'test_fake'))), OpInfo('arange', dtypes=all_types_and(torch.bfloat16, torch.float16), supports_out=True, supports_autograd=False, is_factory_function=True, error_inputs_func=error_inputs_arange, sample_inputs_func=sample_inputs_arange, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestLazyOpInfo', 'test_dispatched_to_lazy'), DecorateInfo(unittest.skip('Skipped!'), 'TestLazyOpInfo', 'test_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestLazyOpInfo', 'test_correctness_with_reusing_ir'), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'))), OpInfo('cauchy', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.cauchy_, inp, *args, **kwargs), inplace_variant=torch.Tensor.cauchy_, dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_cauchy, error_inputs_func=error_inputs_cauchy, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_vmap_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.skip("make_traced() doesn't set seed properly!"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick'))), OpInfo('exponential', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.exponential_, inp, *args, **kwargs), inplace_variant=torch.Tensor.exponential_, dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_exponential, error_inputs_func=error_inputs_exponential, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestVmapOperatorsOpInfo', 'test_vmap_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('geometric', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.geometric_, inp, *args, **kwargs), inplace_variant=torch.Tensor.geometric_, dtypes=floating_types_and(torch.float16, torch.bfloat16, torch.int8, torch.int16, torch.int32, torch.int64, torch.uint8), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_geometric, error_inputs_func=error_inputs_geometric, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_vmap_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick'))), OpInfo('log_normal', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.log_normal_, inp, *args, **kwargs), inplace_variant=torch.Tensor.log_normal_, dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_log_normal, error_inputs_func=error_inputs_log_normal, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_vmap_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick'))), OpInfo('normal', variant_test_name='in_place', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.normal_, inp, *args, **kwargs), inplace_variant=torch.Tensor.normal_, dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_normal, error_inputs_func=error_inputs_normal, skips=(DecorateInfo(unittest.skip('Test expects tensor input'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_vmap_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'))), OpInfo('uniform', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.uniform_, inp, *args, **kwargs), method_variant=None, inplace_variant=torch.Tensor.uniform_, dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16), supports_out=False, supports_autograd=False, is_factory_function=False, sample_inputs_func=sample_inputs_uniform, error_inputs_func=error_inputs_uniform, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), BinaryUfuncInfo('clamp_max', ref=_clamp_max_numpy, dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), supports_forward_ad=True, supports_rhs_python_scalar=False, supports_fwgrad_bwgrad=True, rhs_make_tensor_kwargs=dict(exclude_zero=False), skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestLazyOpInfo', 'test_dispatched_to_lazy'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_errors'))), BinaryUfuncInfo('clamp_min', ref=_clamp_min_numpy, dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), supports_forward_ad=True, supports_rhs_python_scalar=False, supports_fwgrad_bwgrad=True, rhs_make_tensor_kwargs=dict(exclude_zero=False), skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestLazyOpInfo', 'test_dispatched_to_lazy'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_errors'))), BinaryUfuncInfo('mul', aliases=('multiply',), dtypes=all_types_and_complex_and(torch.chalf, torch.float16, torch.bfloat16, torch.bool), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_two_python_scalars=True, error_inputs_sparse_func=error_inputs_sparse_mul, sample_inputs_sparse_coo_func=partial(sample_inputs_sparse_mul, layout=torch.sparse_coo), sample_inputs_sparse_csr_func=partial(sample_inputs_sparse_mul, layout=torch.sparse_csr), sample_inputs_sparse_csc_func=partial(sample_inputs_sparse_mul, layout=torch.sparse_csc), sample_inputs_sparse_bsr_func=partial(sample_inputs_sparse_mul, layout=torch.sparse_bsr), sample_inputs_sparse_bsc_func=partial(sample_inputs_sparse_mul, layout=torch.sparse_bsc)), BinaryUfuncInfo('sub', ref=lambda input, other, *, alpha=1: np.subtract(input, np.multiply(alpha, other)), aliases=('subtract',), dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.chalf), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_add_sub, supports_two_python_scalars=True, decorators=(DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0), torch.bfloat16: tol(atol=1e-05, rtol=0.005), torch.complex32: tol(atol=1e-05, rtol=0.001)}), 'TestBinaryUfuncs', 'test_reference_numerics'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.01, rtol=0)}), 'TestCommon', 'test_complex_half_reference_testing', device_type='cpu'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.005, rtol=0)}), 'TestDecomp', 'test_comprehensive', device_type='cpu'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.005, rtol=0)}), 'TestDecomp', 'test_quick', device_type='cpu')), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.uint8,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.uint8,)))), OpInfo('addmm', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=sample_inputs_addmm, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)),)), OpInfo('addmm', variant_test_name='decomposed', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, autodiff_nonfusible_nodes=['aten::add', 'aten::mm'], sample_inputs_func=partial(sample_inputs_addmm, alpha=1, beta=1), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', device_type='cpu', dtypes=(torch.float16,)))), OpInfo('addmv', dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=[DecorateInfo(toleranceOverride({torch.half: tol(atol=1e-05, rtol=0.003)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cpu')], sample_inputs_func=sample_inputs_addmv), OpInfo('addbmm', ref=lambda M, batch1, batch2, beta=1, alpha=1: np.add(np.multiply(np.asarray(beta, dtype=M.dtype), M), np.multiply(np.asarray(alpha, dtype=batch1.dtype), np.sum(np.matmul(batch1, batch2), axis=0))), dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *([torch.bfloat16] if SM53OrLater or TEST_WITH_ROCM else [])), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=[DecorateInfo(toleranceOverride({torch.float32: tol(atol=1.3e-05, rtol=1.3e-05), torch.complex64: tol(atol=1e-05, rtol=0.0012)}), 'TestCommon', 'test_numpy_refs'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.00013, rtol=0.00013), torch.complex64: tol(atol=1e-05, rtol=0.0012)}), 'TestCommon', 'test_numpy_ref_mps'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-05)}), 'TestConsistency', 'test_output_match'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=1.5e-05, rtol=1e-05)}), 'TestCommon', 'test_out'), DecorateInfo(toleranceOverride({torch.half: tol(atol=0.006, rtol=0.006)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cpu')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes', device_type='cuda', active_if=not SM53OrLater), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager')), sample_inputs_func=sample_inputs_addbmm), OpInfo('baddbmm', dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.float16, torch.complex64, torch.complex128, torch.bfloat16), backward_dtypesIfCUDA=floating_types_and(torch.float16, *([torch.bfloat16] if SM53OrLater or TEST_WITH_ROCM else []), torch.complex64, torch.complex128), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=[DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=0.0012)}), 'TestCommon', 'test_variant_consistency_eager', device_type='cuda'), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=0.0012)}), 'TestMathBits', 'test_conj_view', device_type='cuda')], sample_inputs_func=sample_inputs_baddbmm, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)),)), OpInfo('dot', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), assert_autodiffed=True, sample_inputs_func=sample_inputs_dot_vdot, error_inputs_func=error_inputs_dot_vdot, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)),)), OpInfo('vdot', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_dot_vdot, error_inputs_func=error_inputs_dot_vdot, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)),)), OpInfo('bmm', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *([torch.bfloat16] if SM53OrLater or TEST_WITH_ROCM else [])), assert_autodiffed=True, assert_jit_shape_analysis=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes', device_type='cuda', active_if=not SM53OrLater), DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-05)}), 'TestCommon', 'test_out')), sample_inputs_func=sample_inputs_bmm), OpInfo('mv', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_mv), OpInfo('addr', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16)),), sample_inputs_func=sample_inputs_addr, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL), OpInfo('addcmul', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'),), sample_inputs_func=sample_inputs_addcmul_addcdiv, reference_inputs_func=partial(reference_inputs_elementwise_ternary, sample_inputs_func=reference_inputs_addcmul_addcdiv)), OpInfo('addcdiv', dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'),), sample_inputs_func=sample_inputs_addcmul_addcdiv, reference_inputs_func=partial(reference_inputs_elementwise_ternary, sample_inputs_func=reference_inputs_addcmul_addcdiv)), UnaryUfuncInfo('asin', aliases=('arcsin',), ref=np.arcsin, domain=(-1, 1), supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-05, rtol=0.001)}), 'TestUnaryUfuncs', device_type='cuda'), precisionOverride({torch.bfloat16: 0.01})], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'))), UnaryUfuncInfo('asinh', aliases=('arcsinh',), ref=np.arcsinh, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), decorators=(precisionOverride({torch.bfloat16: 0.05}),), supports_inplace_autograd=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'))), UnaryUfuncInfo('atan', aliases=('arctan',), ref=np.arctan, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', active_if=TEST_WITH_ROCM, device_type='cuda', dtypes=[torch.complex64, torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', active_if=TEST_WITH_ROCM, device_type='cuda', dtypes=[torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'))), BinaryUfuncInfo('atan2', aliases=('arctan2',), dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'),)), UnaryUfuncInfo('atanh', aliases=('arctanh',), ref=np.arctanh, domain=(-1, 1), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), decorators=(precisionOverride({torch.bfloat16: 0.01}),), supports_inplace_autograd=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cfloat], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', active_if=TEST_WITH_ROCM, device_type='cuda', dtypes=[torch.complex128]))), OpInfo('allclose', dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), ref=np.allclose, supports_autograd=False, supports_forward_ad=False, sample_inputs_func=sample_inputs_allclose, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo')), supports_out=False), OpInfo('broadcast_to', ref=np.broadcast_to, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_broadcast_to), OpInfo('broadcast_shapes', op=torch.broadcast_shapes, ref=np.broadcast_shapes if np.lib.NumpyVersion(np.__version__) >= '1.20.0' else None, dtypes=_dispatch_dtypes((torch.float32,)), supports_out=False, supports_gradgrad=False, assert_autodiffed=False, supports_autograd=False, supports_scripting=False, sample_inputs_func=sample_inputs_broadcast_shapes, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'))), OpInfo('broadcast_tensors', ref=np.broadcast_arrays, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_broadcast_tensors, reference_inputs_func=reference_inputs_broadcast_tensors, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32]))), OpInfo('block_diag', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32])), sample_inputs_func=sample_inputs_block_diag), UnaryUfuncInfo('bitwise_not', ref=np.bitwise_not, dtypes=integral_types_and(torch.bool), operator_variant=operator.invert, supports_autograd=False), BinaryUfuncInfo('bitwise_left_shift', op=torch.bitwise_left_shift, dtypes=integral_types(), dtypesIfCUDA=integral_types(), operator_variant=operator.lshift, inplace_operator_variant=operator.ilshift, supports_autograd=False, supports_one_python_scalar=True, rhs_make_tensor_kwargs=dict(low=0), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Some inputs produce undefined outputs'), 'TestCommon', 'test_compare_cpu'))), BinaryUfuncInfo('bitwise_right_shift', op=torch.bitwise_right_shift, dtypes=integral_types(), dtypesIfCUDA=integral_types(), operator_variant=operator.rshift, inplace_operator_variant=operator.irshift, supports_autograd=False, supports_one_python_scalar=True, rhs_make_tensor_kwargs=dict(low=0), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Some inputs produce undefined outputs'), 'TestCommon', 'test_compare_cpu'))), OpInfo('combinations', op=torch.combinations, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, supports_out=False, sample_inputs_func=sample_inputs_combinations), OpInfo('cartesian_prod', op=torch.cartesian_prod, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_cartesian_prod, skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)))), OpInfo('cdist', dtypes=floating_types(), supports_out=False, supports_gradgrad=False, assert_autodiffed=False, sample_inputs_func=sample_inputs_cdist), UnaryUfuncInfo('ceil', ref=np.ceil, dtypes=all_types_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=tuple((t for t in integral_types() if t != torch.uint8))),), supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, assert_autodiffed=True), OpInfo('cholesky', dtypes=floating_and_complex_types(), sample_inputs_func=sample_inputs_linalg_cholesky, gradcheck_wrapper=gradcheck_wrapper_hermitian_input, decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]), OpInfo('cholesky_inverse', dtypes=floating_and_complex_types(), backward_dtypes=floating_and_complex_types(), gradcheck_fast_mode=True, supports_fwgrad_bwgrad=True, supports_forward_ad=True, check_batched_gradgrad=True, sample_inputs_func=sample_inputs_linalg_cholesky_inverse, gradcheck_wrapper=gradcheck_wrapper_triangular_input_real_positive_diagonal, decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack], skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),)), OpInfo('cholesky_solve', op=torch.cholesky_solve, dtypes=floating_and_complex_types(), sample_inputs_func=sample_inputs_cholesky_solve, check_batched_gradgrad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_wrapper=lambda *args, **kwargs: gradcheck_wrapper_triangular_input(*args, idx=1, **kwargs), decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]), OpInfo('chunk', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), sample_inputs_func=sample_inputs_chunk, reference_inputs_func=reference_inputs_chunk, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('unsafe_chunk', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), sample_inputs_func=sample_inputs_chunk, check_batched_forward_grad=False, reference_inputs_func=reference_inputs_chunk, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('clone', ref=np.copy, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), sample_inputs_func=sample_inputs_clone_contiguous, reference_inputs_func=reference_inputs_clone_contiguous, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref_mps'))), OpInfo('contiguous', op=lambda x, *args, **kwargs: x.contiguous(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), sample_inputs_func=sample_inputs_clone_contiguous, reference_inputs_func=reference_inputs_clone_contiguous, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_fusible_nodes=['aten::contiguous'], assert_jit_shape_analysis=True, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), OpInfo('sum_to_size', op=lambda x, *args, **kwargs: x.sum_to_size(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_sum_to_size, error_inputs_func=error_inputs_sum_to_size, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float,)))), OpInfo('clamp', aliases=('clip',), ref=_clamp_numpy, dtypes=all_types_and(torch.bfloat16, torch.half), sample_inputs_func=sample_inputs_clamp, reference_inputs_func=partial(reference_inputs_elementwise_ternary, sample_inputs_func=sample_inputs_clamp), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bool,)),)), UnaryUfuncInfo('positive', ref=np.positive, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True), UnaryUfuncInfo('conj', ref=np.conj, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), supports_sparse=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, supports_out=False), UnaryUfuncInfo('conj_physical', decomp_aten_name='_conj_physical', ref=np.conj, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.skip('Skipped! conj_physical_ not implemented for sparse'), 'TestSparseUnaryUfuncs', 'test_inplace'))), OpInfo('resolve_conj', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_view_as_real, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('resolve_neg', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_view_as_real, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('view_as_real', dtypes=complex_types(), supports_forward_ad=True, supports_out=False, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_view_as_real, test_conjugated_samples=False), OpInfo('view_as_complex', dtypes=floating_types_and(torch.half), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, test_neg_view=False, sample_inputs_func=sample_inputs_view_as_complex, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.half,)), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), BinaryUfuncInfo('complex', dtypes=floating_types_and(torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_rhs_python_scalar=False, error_inputs_func=error_inputs_complex, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out', device_type='mps'))), BinaryUfuncInfo('copysign', dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), promotes_int_to_float=True, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True), OpInfo('corrcoef', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_corrcoef, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)),), supports_out=False), UnaryUfuncInfo('cos', ref=np.cos, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, handles_large_floats=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cfloat, torch.cdouble), device_type='cpu', active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cdouble,), device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=(torch.chalf,), active_if=IS_WINDOWS))), UnaryUfuncInfo('cosh', ref=np_unary_ufunc_integer_promotion_wrapper(np.cosh), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.int8]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=(torch.chalf,), active_if=IS_WINDOWS))), OpInfo('cov', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_cov, error_inputs_func=error_inputs_cov, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.skip('Barely fails'), 'TestFwdGradients', 'test_fn_fwgrad_bwgrad'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), OpInfo('cross', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_cross, supports_fwgrad_bwgrad=True, supports_out=True, supports_forward_ad=True), OpInfo('cumsum', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),), sample_inputs_func=sample_inputs_cumulative_ops), OpInfo('cumprod', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),), sample_inputs_func=sample_inputs_cumprod, gradcheck_fast_mode=False), OpInfo('cummax', dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL), OpInfo('cummin', dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_cumulative_ops, supports_dtype_kwargs=False), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL), UnaryUfuncInfo('deg2rad', ref=np.radians, decorators=(precisionOverride({torch.bfloat16: 0.7, torch.float16: 0.7}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True), OpInfo('diff', op=torch.diff, ref=lambda input, n=1, dim=-1, prepend=np._NoValue, append=np._NoValue: np.diff(input, n, dim, np._NoValue if prepend is None else prepend, np._NoValue if append is None else append), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_diff, error_inputs_func=error_inputs_diff, check_batched_forward_grad=False, skips=()), BinaryUfuncInfo('div', aliases=('divide',), variant_test_name='no_rounding_mode', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), gradcheck_fast_mode=True, supports_forward_ad=True, promotes_int_to_float=True, supports_fwgrad_bwgrad=True, supports_two_python_scalars=True, assert_autodiffed=True, rhs_make_tensor_kwargs=dict(exclude_zero=True)), BinaryUfuncInfo('div', aliases=('divide',), variant_test_name='trunc_rounding', dtypes=all_types_and(torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_elementwise_binary, sample_kwargs=dict(rounding_mode='trunc')), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_two_python_scalars=True, assert_autodiffed=True, rhs_make_tensor_kwargs=dict(exclude_zero=True), decorators=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_working'),)), BinaryUfuncInfo('div', aliases=('divide',), variant_test_name='floor_rounding', dtypes=all_types_and(torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_elementwise_binary, sample_kwargs=dict(rounding_mode='floor')), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_two_python_scalars=True, assert_autodiffed=True, rhs_make_tensor_kwargs=dict(exclude_zero=True), decorators=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_working'),)), BinaryUfuncInfo('true_divide', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_forward_ad=True, promotes_int_to_float=True, supports_fwgrad_bwgrad=True, supports_two_python_scalars=True, rhs_make_tensor_kwargs=dict(exclude_zero=True)), OpInfo('equal', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), ref=lambda input, other: (input == other).all(), sample_inputs_func=sample_inputs_equal, supports_autograd=False, supports_tracing=False, skips=()), UnaryUfuncInfo('exp', ref=np_unary_ufunc_integer_promotion_wrapper(np.exp), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble])), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True), OpInfo('expand', op=lambda self, shape: self.expand(shape), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_expand, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), OpInfo('expand_as', op=lambda self, other: self.expand_as(other), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_expand_as, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), OpInfo('diag', ref=np.diag, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_diag, error_inputs_func=error_inputs_diag), OpInfo('diag_embed', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_out=False, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_diagonal_diag_embed, reference_inputs_func=reference_inputs_diagonal_diag_embed, error_inputs_func=error_inputs_diagonal_diag_embed), OpInfo('diagonal', aten_backward_name='diagonal_backward', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_diagonal_diag_embed, reference_inputs_func=reference_inputs_diagonal_diag_embed, error_inputs_func=error_inputs_diagonal_diag_embed), OpInfo('diagonal_copy', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_diagonal_diag_embed, reference_inputs_func=reference_inputs_diagonal_diag_embed, error_inputs_func=error_inputs_diagonal_diag_embed), OpInfo('diagonal_scatter', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_diagonal_scatter), BinaryUfuncInfo('eq', ref=np.equal, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), always_returns_bool=True, supports_autograd=False, sample_inputs_func=sample_inputs_comparison_ops, skips=()), BinaryUfuncInfo('fmax', op=torch.fmax, dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'),)), BinaryUfuncInfo('fmin', op=torch.fmin, dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'),)), BinaryUfuncInfo('fmod', ref=np.fmod, dtypes=all_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=None, rhs_make_tensor_kwargs={'exclude_zero': True}, decorators=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_contig_vs_every_other', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_non_contig', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.uint8,)))), BinaryUfuncInfo('remainder', ref=np.remainder, dtypes=all_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=None, operator_variant=operator.mod, inplace_operator_variant=operator.imod, supports_one_python_scalar=True, rhs_make_tensor_kwargs={'exclude_zero': True}, decorators=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_contig_vs_every_other', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_non_contig', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.uint8,)), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestOpInfo', device_type='xla', dtypes=(torch.long,)))), UnaryUfuncInfo('frac', ref=lambda x: np.modf(x)[0], dtypes=floating_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.bfloat16, torch.float16, torch.float32, torch.float64)), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bfloat16, torch.float32, torch.float64)))), OpInfo('stft', decorators=[skipCPUIfNoFFT, DecorateInfo(unittest.skip('Skipped! stft does not match the native function'), 'TestJit', 'test_variant_consistency_jit')], dtypes=floating_and_complex_types(), sample_inputs_func=sample_inputs_stft, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, check_batched_grad=False, check_batched_gradgrad=False, supports_out=False, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL), OpInfo('istft', dtypes=complex_types(), sample_inputs_func=sample_inputs_istft, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, check_batched_grad=False, check_batched_gradgrad=False, supports_out=False, decorators=(DecorateInfo(unittest.skip('Skipped! istft does not match the native function'), 'TestJit', 'test_variant_consistency_jit'),), skips=(skipCPUIfNoFFT, DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_backward'))), UnaryUfuncInfo('floor', ref=np.floor, dtypes=all_types_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=tuple((t for t in integral_types() if t != torch.uint8))),), supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, assert_autodiffed=True), OpInfo('flip', op=torch.flip, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_flip, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('fliplr', op=torch.fliplr, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_fliplr_flipud, error_inputs_func=error_inputs_fliplr, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('flipud', op=torch.flipud, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_fliplr_flipud, error_inputs_func=error_inputs_flipud, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('sparse.sampled_addmm', dtypes=floating_and_complex_types(), supports_autograd=True, sample_inputs_func=sample_inputs_sparse_sampled_addmm, decorators=[skipCUDAIf(not (_get_torch_cuda_version() >= (11, 3) or _get_torch_rocm_version() >= (5, 2)), 'cusparseSDDMM was added in 11.2.1'), skipCPUIfNoMklSparse], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Skipped!'), 'TestTags', 'test_tags'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_backward'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_fn_fwgrad_bwgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD'))), OpInfo('sparse.mm', dtypes=floating_types_and(torch.bfloat16), variant_test_name='reduce', supports_autograd=True, supports_out=False, supports_gradgrad=False, supports_forward_ad=False, sample_inputs_func=sample_inputs_sparse_mm_reduce, decorators=[onlyCPU], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestTags', 'test_tags'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_backward'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_fn_fwgrad_bwgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_fail_gradgrad'))), UnaryUfuncInfo('i0', ref=np_unary_ufunc_integer_promotion_wrapper(scipy.special.i0) if TEST_SCIPY else None, aliases=('special.i0',), decorators=(precisionOverride({torch.bfloat16: 0.3, torch.float16: 0.5}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), backward_dtypes=floating_types(), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, sample_inputs_func=sample_inputs_i0_i1, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.int8,)),)), BinaryUfuncInfo('floor_divide', ref=_floor_divide_np, dtypes=all_types_and(torch.half, torch.bfloat16), supports_autograd=False, rhs_make_tensor_kwargs=dict(exclude_zero=True), supports_two_python_scalars=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.int8,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=(torch.float16,)), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.005)}), 'TestBinaryUfuncs', 'test_reference_numerics'))), UnaryUfuncInfo('frexp', op=torch.frexp, ref=np.frexp, dtypes=floating_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half), decorators=[], supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_batch_vs_slicing'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_contig_vs_every_other'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_contig_vs_transposed'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_non_contig_expand'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_variant_consistency'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_out_arg_all_dtypes'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', active_if=IS_WINDOWS))), UnaryUfuncInfo('log1p', ref=np.log1p, aliases=('special.log1p',), domain=(-1, None), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), decorators=(precisionOverride({torch.bfloat16: 0.1}),), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, assert_autodiffed=True, promotes_int_to_float=True), BinaryUfuncInfo('ge', ref=np.greater_equal, aliases=('greater_equal',), dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16), always_returns_bool=True, supports_autograd=False, skips=()), OpInfo('geqrf', dtypes=floating_and_complex_types(), sample_inputs_func=sample_inputs_linalg_qr_geqrf, decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack], supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'))), BinaryUfuncInfo('gt', ref=np.greater, aliases=('greater',), dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16), always_returns_bool=True, supports_autograd=False, skips=()), UnaryUfuncInfo('imag', ref=np.imag, dtypes=complex_types_and(torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_out_arg_all_dtypes'),)), OpInfo('gradient', dtypes=floating_and_complex_types_and(torch.int8, torch.int16, torch.int32, torch.int64, torch.bfloat16, torch.half), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo')), supports_inplace_autograd=False, sample_inputs_func=sample_inputs_gradient, error_inputs_func=error_inputs_gradient), OpInfo('isin', dtypes=all_types(), dtypesIfCUDA=all_types_and(torch.half), supports_autograd=False, sample_inputs_func=sample_inputs_isin), OpInfo('kthvalue', dtypes=all_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and(torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_kthvalue, error_inputs_func=error_inputs_kthvalue), BinaryUfuncInfo('le', ref=np.less_equal, aliases=('less_equal',), dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16), always_returns_bool=True, supports_autograd=False, skips=()), OpInfo('linspace', dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16), is_factory_function=True, supports_out=True, supports_autograd=False, error_inputs_func=error_inputs_linspace, sample_inputs_func=sample_inputs_linspace, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.cfloat,), device_type='cuda'))), OpInfo('linspace', dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16), is_factory_function=True, supports_out=True, supports_autograd=False, error_inputs_func=error_inputs_linspace, sample_inputs_func=sample_inputs_linspace_tensor_overload, variant_test_name='tensor_overload', skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.cfloat,), device_type='cuda'))), OpInfo('logspace', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), is_factory_function=True, supports_out=True, supports_autograd=False, error_inputs_func=error_inputs_linspace, sample_inputs_func=sample_inputs_logspace, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_comprehensive', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.cfloat,), device_type='cuda'))), OpInfo('logspace', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), is_factory_function=True, supports_out=True, supports_autograd=False, error_inputs_func=error_inputs_linspace, sample_inputs_func=sample_inputs_logspace_tensor_overload, variant_test_name='tensor_overload', skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_comprehensive', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.cfloat,), device_type='cuda'))), UnaryUfuncInfo('log', ref=np.log, domain=(0, None), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16, torch.chalf), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.bfloat16: 0.05}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),), reference_numerics_filter=NumericsFilter(condition=lambda x: torch.abs(x) < 0.1, safe_val=1)), UnaryUfuncInfo('log10', ref=np.log10, domain=(0, None), decorators=(precisionOverride({torch.bfloat16: 0.05}),), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),), reference_numerics_filter=NumericsFilter(condition=lambda x: torch.abs(x) < 0.1, safe_val=1)), UnaryUfuncInfo('log2', ref=np.log2, domain=(0, None), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.bfloat16: 0.1}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble]),), reference_numerics_filter=NumericsFilter(condition=lambda x: torch.abs(x) < 0.1, safe_val=1)), BinaryUfuncInfo('ldexp', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_inplace_autograd=False, promotes_int_to_float=True, supports_out=True, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view')), decorators=[DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=1e-05)}), 'TestCommon', device_type='cpu')]), BinaryUfuncInfo('logaddexp', dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16), dtypesIfROCM=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'),)), OpInfo('logaddexp2', dtypes=floating_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=floating_types_and(torch.bfloat16), dtypesIfROCM=floating_types_and(torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_logaddexp), UnaryUfuncInfo('logical_not', ref=np.logical_not, decorators=(precisionOverride({torch.bfloat16: 0.7, torch.float16: 0.5}),), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_autograd=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_variant_consistency', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16)))), BinaryUfuncInfo('lt', ref=np.less, aliases=('less',), dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16), always_returns_bool=True, supports_autograd=False, skips=()), OpInfo('lu_unpack', op=torch.lu_unpack, dtypes=floating_and_complex_types(), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(skipCPUIfNoLapack,), sample_inputs_func=sample_inputs_lu_unpack), OpInfo('lu', op=torch.lu, dtypes=floating_and_complex_types(), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_lu, decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack], skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'))), OpInfo('lu_solve', op=torch.lu_solve, dtypes=floating_and_complex_types(), supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_lu_solve, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Tests different backward paths'), 'TestCommon', 'test_floating_inputs_are_differentiable')), decorators=[skipCPUIfNoLapack, skipCUDAIfNoMagmaAndNoCusolver]), OpInfo('masked_fill', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_masked_fill, error_inputs_func=error_inputs_masked_fill, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, supports_out=False), OpInfo('masked_scatter', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_masked_scatter, error_inputs_func=error_inputs_masked_scatter, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, supports_out=False, skips=()), OpInfo('masked_select', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_masked_select, error_inputs_func=error_inputs_masked_select, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values', dtypes=[torch.bool], active_if=TEST_WITH_ROCM),)), OpInfo('matrix_exp', dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), aliases=('linalg.matrix_exp',), sample_inputs_func=sample_inputs_matrix_exp, check_batched_grad=False, check_batched_gradgrad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestInductorOpInfo', 'test_comprehensive', dtypes=[torch.half], device_type='cpu'),), supports_out=False), OpInfo('matmul', aliases=('linalg.matmul',), dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *([torch.bfloat16] if SM53OrLater or TEST_WITH_ROCM else [])), assert_autodiffed=True, assert_jit_shape_analysis=True, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=partial(sample_inputs_matmul, is_rmatmul=False), decorators=[DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes', device_type='cuda', active_if=not SM53OrLater), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=0)}), 'TestCommon', 'test_noncontiguous_samples', device_type='cuda', active_if=TEST_WITH_ROCM), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=0)}), 'TestCommon', 'test_out', device_type='cuda', active_if=TEST_WITH_ROCM), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0, rtol=1e-05)}), 'TestCommon', 'test_noncontiguous_samples', device_type='cpu'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=1e-05), torch.complex64: tol(atol=1e-05, rtol=1e-05)}), 'TestDecomp', 'test_comprehensive', device_type='cuda')], skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('67470!'), 'TestCommon', 'test_noncontiguous_samples', device_type='cpu', dtypes=(torch.long,)), DecorateInfo(unittest.skip('Skipped!'), 'TestOpInfo', device_type='xla', dtypes=(torch.long,)), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', device_type='cpu', dtypes=(torch.long,)))), OpInfo('max', variant_test_name='reduction_with_dim', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), sample_inputs_func=sample_inputs_max_min_reduction_with_dim, supports_fwgrad_bwgrad=True, skips=(), supports_forward_ad=True), OpInfo('max', variant_test_name='reduction_no_dim', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_out=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_max_min_reduction_no_dim, skips=()), OpInfo('median', dtypes=all_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and(torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, error_inputs_func=error_inputs_median, sample_inputs_func=partial(sample_inputs_reduction, supports_multiple_dims=False)), OpInfo('nanmedian', dtypes=all_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and(torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=partial(sample_inputs_reduction, supports_multiple_dims=False)), OpInfo('var_mean', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var, supports_out=False, supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(toleranceOverride({torch.float64: tol(atol=2e-07, rtol=2e-07)}), 'TestDecomp', 'test_comprehensive', device_type='cuda'),)), OpInfo('var_mean', variant_test_name='unbiased', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var_unbiased, supports_out=False, supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(toleranceOverride({torch.float64: tol(atol=2e-07, rtol=2e-07)}), 'TestDecomp', 'test_comprehensive', device_type='cuda'),)), OpInfo('std_mean', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var, supports_out=False, supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(toleranceOverride({torch.float64: tol(atol=2e-07, rtol=2e-07)}), 'TestDecomp', 'test_comprehensive', device_type='cuda'),)), OpInfo('std_mean', variant_test_name='unbiased', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var_unbiased, supports_out=False, supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(toleranceOverride({torch.float64: tol(atol=2e-07, rtol=2e-07)}), 'TestDecomp', 'test_comprehensive', device_type='cuda'),)), OpInfo('meshgrid', variant_test_name='variadic_tensors', ref=np.meshgrid, dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool, torch.float16), sample_inputs_func=partial(sample_inputs_meshgrid, variant='variadic'), skips=[DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive')], supports_out=False, supports_fwgrad_bwgrad=True, supports_forward_ad=True, check_batched_forward_grad=False), OpInfo('meshgrid', variant_test_name='list_of_tensors', dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool, torch.float16), sample_inputs_func=partial(sample_inputs_meshgrid, variant='list'), skips=[DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive')], assert_autodiffed=True, supports_out=False, autodiff_nonfusible_nodes=[], supports_fwgrad_bwgrad=True, supports_forward_ad=True, check_batched_forward_grad=False), OpInfo('min', variant_test_name='reduction_with_dim', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), sample_inputs_func=sample_inputs_max_min_reduction_with_dim, supports_fwgrad_bwgrad=True, supports_forward_ad=True, skips=()), OpInfo('min', variant_test_name='reduction_no_dim', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_out=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_max_min_reduction_no_dim, skips=()), OpInfo('quantile', dtypes=floating_types(), sample_inputs_func=sample_inputs_reduction_quantile, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False), OpInfo('nanquantile', dtypes=floating_types(), sample_inputs_func=sample_inputs_reduction_quantile, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False), BinaryUfuncInfo('max', aliases=('maximum',), variant_test_name='binary', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, ref=np.maximum, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'), DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'))), BinaryUfuncInfo('maximum', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, ref=np.maximum, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'),)), BinaryUfuncInfo('min', aliases=('minimum',), variant_test_name='binary', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, ref=np.minimum, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'), DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'))), BinaryUfuncInfo('minimum', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, ref=np.minimum, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'),)), BinaryUfuncInfo('logical_and', ref=np.logical_and, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_autograd=False, always_returns_bool=True, supports_rhs_python_scalar=False), BinaryUfuncInfo('logical_or', ref=np.logical_or, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_autograd=False, always_returns_bool=True, supports_rhs_python_scalar=False), BinaryUfuncInfo('logical_xor', ref=np.logical_xor, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_autograd=False, always_returns_bool=True, supports_rhs_python_scalar=False, skips=()), BinaryUfuncInfo('bitwise_and', ref=np.bitwise_and, dtypes=integral_types_and(torch.bool), operator_variant=operator.and_, inplace_operator_variant=operator.iand, supports_autograd=False, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'),)), BinaryUfuncInfo('bitwise_or', ref=np.bitwise_or, dtypes=integral_types_and(torch.bool), operator_variant=operator.or_, inplace_operator_variant=operator.ior, supports_autograd=False, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'),)), BinaryUfuncInfo('bitwise_xor', ref=np.bitwise_xor, dtypes=integral_types_and(torch.bool), operator_variant=operator.xor, inplace_operator_variant=operator.ixor, supports_autograd=False, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion', device_type='cuda'),)), BinaryUfuncInfo('heaviside', ref=lambda a, b: np.int64(np.heaviside(a, b)) if a.dtype == np.int64 and b.dtype == np.int64 else np.heaviside(a, b), dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), supports_autograd=False, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values'))), BinaryUfuncInfo('lcm', ref=np.lcm, dtypes=integral_types_and(), supports_autograd=False, supports_rhs_python_scalar=False), BinaryUfuncInfo('gcd', ref=np.gcd, dtypes=integral_types_and(), supports_autograd=False, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.int8,)),)), BinaryUfuncInfo('isclose', ref=np.isclose, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_isclose, error_inputs_func=error_inputs_isclose, supports_autograd=False, supports_out=False, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_refs', dtypes=(torch.complex128,)), DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values'))), OpInfo('softmax', aliases=('special.softmax', 'nn.functional.softmax'), aten_name='softmax', aten_backward_name='_softmax_backward_data', dtypes=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_softmax_variant, assert_jit_shape_analysis=True, assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=True), OpInfo('softmax', aliases=('special.softmax', 'nn.functional.softmax'), variant_test_name='with_dtype', aten_name='softmax', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=partial(sample_inputs_softmax_variant, with_dtype=True), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=True), OpInfo('_softmax_backward_data', op=torch.ops.aten._softmax_backward_data, aten_name='_softmax_backward_data', dtypes=floating_types_and(torch.bfloat16, torch.float16), sample_inputs_func=sample_inputs_softmax_backward_data, assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples', device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)))), OpInfo('nn.functional.softmin', aten_name='softmin', dtypes=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_softmax_variant, assert_jit_shape_analysis=False, assert_autodiffed=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('nn.functional.softmin', variant_test_name='with_dtype', aten_name='softmin', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), sample_inputs_func=partial(sample_inputs_softmax_variant, with_dtype=True), assert_autodiffed=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('nn.functional.cross_entropy', dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_cross_entropy, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=0.001)}), 'TestJit', 'test_variant_consistency_jit', device_type='cpu'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', device_type='cuda'),)), OpInfo('nn.functional.normalize', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_normalize, supports_forward_ad=True, supports_fwgrad_bwgrad=True), OpInfo('aminmax', ref=lambda x, dim=None, keepdim=False: (np.amin(x, axis=dim, keepdims=keepdim), np.amax(x, axis=dim, keepdims=keepdim)), dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), decorators=(onlyNativeDeviceTypes,), supports_autograd=False, sample_inputs_func=sample_inputs_aminmax, error_inputs_func=error_inputs_aminmax_amax_amin), OpInfo('as_strided', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_inplace_batched_forward_grad=False, sample_inputs_func=sample_inputs_as_strided, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Numerous errors'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Numerous errors'), 'TestBwdGradients'))), OpInfo('as_strided', variant_test_name='partial_views', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_inplace_batched_forward_grad=False, sample_inputs_func=sample_inputs_as_strided_partial_views, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_fn_fwgrad_bwgrad', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_inplace_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_inplace_grad'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_inplace_gradgrad'), DecorateInfo(unittest.expectedFailure, 'TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive_inplace'), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Test changes in memory layout'), 'TestMathBits'), DecorateInfo(unittest.skip('Modifies input strides and storage_offset'), 'TestCommon', 'test_non_standard_bool_values'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace_all_strides'))), OpInfo('as_strided_scatter', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_inplace_batched_forward_grad=False, sample_inputs_func=sample_inputs_as_strided_scatter, error_inputs_func=error_inputs_as_strided_scatter, skips=(DecorateInfo(unittest.skip('Works for int64, fails for everything else'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Fails in most cases, passes on LAZY for some reason'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Fails on cuda + rocm'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.skip('Passes on complex128 and float64 only'), 'TestFwdGradients', 'test_fn_fwgrad_bwgrad'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestDecomp', 'test_comprehensive'))), OpInfo('native_layer_norm', aten_name='native_layer_norm', ref=reference_native_layer_norm, dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, assert_jit_shape_analysis=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_native_layer_norm, error_inputs_func=error_inputs_native_layer_norm, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Unsupported on MPS for now'), 'TestCommon', 'test_numpy_ref_mps'))), OpInfo('native_batch_norm', aten_name='native_batch_norm', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, sample_inputs_func=sample_inputs_native_batch_norm, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning', device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestFakeTensor', 'test_fake_autocast'), DecorateInfo(unittest.skip('Skipped!'), 'TestFakeTensor', 'test_fake'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=5e-05, rtol=5e-05)}), 'TestCompositeCompliance', 'test_forward_ad'))), OpInfo('_native_batch_norm_legit', aten_name='_native_batch_norm_legit', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, sample_inputs_func=sample_inputs__native_batch_norm_legit, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning', device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_compare_cpu'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=5e-05, rtol=5e-05)}), 'TestCompositeCompliance', 'test_forward_ad'))), OpInfo('nn.functional.cosine_similarity', aten_name='cosine_similarity', dtypes=floating_types_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_cosine_similarity), OpInfo('nn.functional.adaptive_avg_pool1d', dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_adaptive_avg_pool1d, sample_inputs_func=sample_inputs_adaptive_avg_pool1d), OpInfo('nn.functional.adaptive_avg_pool2d', dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), decorators=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_adaptive_avg_pool2d, sample_inputs_func=sample_inputs_adaptive_avg_pool2d), OpInfo('nn.functional.adaptive_avg_pool3d', dtypes=floating_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), decorators=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_adaptive_avg_pool3d, sample_inputs_func=sample_inputs_adaptive_avg_pool3d), OpInfo('nn.functional.adaptive_max_pool1d', dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_adaptive_max_pool1d, sample_inputs_func=sample_inputs_adaptive_max_pool1d), OpInfo('nn.functional.adaptive_max_pool2d', dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), decorators=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_adaptive_max_pool2d, sample_inputs_func=sample_inputs_adaptive_max_pool2d), OpInfo('nn.functional.adaptive_max_pool3d', dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), decorators=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_adaptive_max_pool3d, sample_inputs_func=sample_inputs_adaptive_max_pool3d), OpInfo('nn.functional.avg_pool1d', aten_name='avg_pool1d', supports_autograd=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.int64, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_avg_pool1d, sample_inputs_func=sample_inputs_avgpool1d), OpInfo('nn.functional.avg_pool3d', aten_name='avg_pool3d', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.int64), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_avg_pool3d, sample_inputs_func=sample_inputs_avgpool3d, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cpu'),)), OpInfo('nn.functional.binary_cross_entropy_with_logits', aten_name='binary_cross_entropy_with_logits', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, dtypes=floating_types_and(torch.half, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=sample_inputs_binary_cross_entropy_with_logits, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),)), UnaryUfuncInfo('nn.functional.relu', aten_name='relu', ref=lambda a: np.where(a <= 0, 0, a), supports_autograd=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, dtypes=all_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_nn_activation_relu, supports_out=False, supports_fwgrad_bwgrad=True, supports_forward_ad=True), OpInfo('nn.functional.conv_transpose1d', ref=partial(conv_transpose_ref, fn=torch.nn.functional.conv_transpose1d), aten_name='conv_transpose1d', aliases=('conv_transpose1d',), dtypes=floating_and_complex_types_and(torch.int64, torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.chalf, torch.bfloat16), sample_inputs_func=sample_inputs_conv_transpose1d, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, decorators=(DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=1.3e-06)}), 'TestCommon', 'test_variant_consistency_eager', device_type='cuda'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.05, rtol=0.05)}), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(toleranceOverride({torch.float: tol(atol=1.5e-05, rtol=1.5e-05)}), 'TestCommon', 'test_numpy_ref_mps'), DecorateInfo(toleranceOverride({torch.half: tol(atol=0.001, rtol=0.002)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cpu')), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.complex64,)), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref', dtypes=(torch.int64,))), supports_out=False), OpInfo('nn.functional.conv_transpose2d', aten_name='conv_transpose2d', aliases=('conv_transpose2d',), ref=partial(conv_transpose_ref, fn=torch.nn.functional.conv_transpose2d), dtypes=floating_and_complex_types_and(torch.int64, torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.chalf, torch.bfloat16), sample_inputs_func=sample_inputs_conv_transpose2d, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, decorators=[DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=1.3e-06)}), 'TestCommon', 'test_variant_consistency_eager', device_type='cuda'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=2e-05, rtol=5e-05)}), 'TestCommon', 'test_noncontiguous_samples', device_type='cuda'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.08, rtol=0.08)}), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(toleranceOverride({torch.half: tol(atol=0.001, rtol=0.002)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cpu')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref', dtypes=(torch.int64,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref', dtypes=(torch.double, torch.cdouble)), DecorateInfo(unittest.skip('Unsupported on MPS for now'), 'TestCommon', 'test_numpy_ref_mps'), DecorateInfo(unittest.expectedFailure, 'TestDtypeCustomRules', 'test_custom_rules', dtypes=(torch.complex64, torch.complex128))), supports_out=False), OpInfo('nn.functional.conv_transpose3d', aten_name='conv_transpose3d', aliases=('conv_transpose3d',), ref=partial(conv_transpose_ref, fn=torch.nn.functional.conv_transpose3d), dtypes=floating_and_complex_types_and(torch.int64, torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.chalf, torch.bfloat16), sample_inputs_func=sample_inputs_conv_transpose3d, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, gradcheck_fast_mode=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.05, rtol=0.05)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cuda'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=1.3e-06), torch.complex64: tol(atol=0.00013, rtol=1.3e-05)}), 'TestCommon', 'test_variant_consistency_eager', device_type='cuda'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0002, rtol=0.0002)}), 'TestCompositeCompliance', 'test_operator', device_type='cuda'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.00013, rtol=1.3e-06), torch.complex64: tol(atol=0.00013, rtol=1.3e-05)}), 'TestCommon', 'test_noncontiguous_samples', device_type='cuda'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=2e-05)}), 'TestCompositeCompliance', 'test_forward_ad', device_type='cuda', active_if=TEST_CUDNN), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=0.0001)}), 'TestMathBits', 'test_conj_view', device_type='cuda'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.09, rtol=0.09)}), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(toleranceOverride({torch.half: tol(atol=0.001, rtol=0.2)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cpu')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref', dtypes=(torch.int64,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref', dtypes=(torch.double, torch.cdouble)), DecorateInfo(unittest.skip('Unsupported on MPS for now'), 'TestCommon', 'test_numpy_ref_mps'), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.complex64, torch.complex128))), supports_out=False), OpInfo('nn.functional.conv1d', aliases=('conv1d',), aten_name='conv1d', dtypes=floating_and_complex_types_and(torch.int64, torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.chalf, torch.bfloat16), sample_inputs_func=sample_inputs_conv1d, error_inputs_func=error_inputs_conv1d, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, decorators=(DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.01, rtol=0.05)}), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.002, rtol=0.001)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cuda')), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestDtypeCustomRules', 'test_custom_rules', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.complex64, torch.complex128))), supports_expanded_weight=True, supports_out=False), OpInfo('nn.functional.conv2d', aliases=('conv2d',), aten_name='conv2d', dtypes=floating_and_complex_types_and(torch.int64, torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.chalf, torch.bfloat16), sample_inputs_func=partial(sample_inputs_conv2d), error_inputs_func=error_inputs_conv2d, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, decorators=(DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.06, rtol=0.05)}), 'TestCommon', 'test_complex_half_reference_testing'),), skips=(DecorateInfo(unittest.skip('Works on some configs!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestDtypeCustomRules', 'test_custom_rules', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.complex64, torch.complex128))), supports_expanded_weight=True, supports_out=False), OpInfo('nn.functional.group_norm', aten_name='group_norm', aliases=('group_norm',), ref=reference_group_norm, dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, error_inputs_func=error_inputs_group_norm, decorators=[DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,))], sample_inputs_func=sample_inputs_group_norm, reference_inputs_func=reference_inputs_group_norm, supports_expanded_weight=True), OpInfo('nn.functional.instance_norm', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=[DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_forward_ad', active_if=TEST_WITH_ROCM)], sample_inputs_func=sample_inputs_instance_norm, supports_expanded_weight=True), OpInfo('nn.functional.layer_norm', aten_name='layer_norm', aten_backward_name='layer_norm_backward', aliases=('layer_norm',), ref=reference_layer_norm, dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, decorators=[DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=0.001)}), 'TestCommon', 'test_numpy_refs'), DecorateInfo(unittest.skip('Bug in MPS backend!'), 'TestCommon', 'test_numpy_ref_mps')], sample_inputs_func=sample_inputs_layer_norm, supports_expanded_weight=True), OpInfo('nn.functional.local_response_norm', dtypes=floating_types_and(torch.int64, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=[DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,))], sample_inputs_func=sample_inputs_local_response_norm), OpInfo('constant_pad_nd', supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half), sample_inputs_func=sample_inputs_constant_pad_nd, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bool,)),)), OpInfo('nn.functional.pad', variant_test_name='constant', aten_name='constant_pad_nd', gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half), sample_inputs_func=partial(sample_inputs_nn_pad, mode='constant'), supports_out=False), OpInfo('nn.functional.pad', variant_test_name='reflect', supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=all_types_and_complex_and(torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_nn_pad, mode='reflect'), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, supports_out=False), OpInfo('nn.functional.pad', variant_test_name='replicate', supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=all_types_and_complex_and(torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_nn_pad, mode='replicate'), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, supports_out=False), OpInfo('nn.functional.pad', variant_test_name='circular', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half), sample_inputs_func=partial(sample_inputs_nn_pad, mode='circular'), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_grad=False, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestDecomp', 'test_comprehensive')), supports_out=False), OpInfo('nn.functional.hardswish', aten_name='hardswish', aten_backward_name='hardswish_backward', supports_autograd=True, assert_autodiffed=True, sample_inputs_func=sample_inputs_hardswish, dtypes=floating_types_and(torch.bfloat16, torch.half), supports_gradgrad=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, autodiff_nonfusible_nodes=['aten::hardswish']), OpInfo('nn.functional.unfold', aten_name='im2col', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_nn_unfold, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, skips=(DecorateInfo(unittest.skip('Internal assert failed!'), 'TestJit', 'test_variant_consistency_jit'),)), OpInfo('nn.functional.interpolate', aten_name='interpolate', variant_test_name='nearest', supports_autograd=True, supports_fwgrad_bwgrad=True, supports_forward_ad=True, dtypes=floating_types_and(torch.uint8, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16, torch.uint8), sample_inputs_func=partial(sample_inputs_interpolate, 'nearest'), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('nn.functional.interpolate', aten_name='interpolate', variant_test_name='nearest-exact', supports_autograd=True, supports_fwgrad_bwgrad=True, supports_forward_ad=True, dtypes=floating_types_and(torch.uint8), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16, torch.uint8), sample_inputs_func=partial(sample_inputs_interpolate, 'nearest-exact'), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestOperators', 'test_vmapjvpall_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestOperators', 'test_vmapvjp_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestInductorOpInfo', 'test_comprehensive'), DecorateInfo(unittest.expectedFailure, 'TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestEagerFusionOpInfo', 'test_aot_autograd_symbolic_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestConsistency')), supports_out=False), OpInfo('nn.functional.interpolate', aten_name='interpolate', variant_test_name='linear', supports_autograd=True, supports_fwgrad_bwgrad=True, supports_forward_ad=True, dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_interpolate, 'linear'), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('nn.functional.interpolate', aten_name='interpolate', variant_test_name='bilinear', supports_fwgrad_bwgrad=True, supports_autograd=True, supports_forward_ad=True, dtypes=floating_types_and(torch.uint8, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=partial(sample_inputs_interpolate, 'bilinear'), reference_inputs_func=partial(reference_inputs_interpolate, 'bilinear'), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('nn.functional.interpolate', aten_name='interpolate', variant_test_name='bicubic', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.uint8, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_interpolate, 'bicubic'), reference_inputs_func=partial(reference_inputs_interpolate, 'bicubic'), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('nn.functional.interpolate', aten_name='interpolate', variant_test_name='trilinear', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=partial(sample_inputs_interpolate, 'trilinear'), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('nn.functional.interpolate', aten_name='interpolate', variant_test_name='area', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=partial(sample_inputs_interpolate, 'area'), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('nn.functional.upsample_bilinear', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.uint8, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=partial(sample_inputs_upsample, 'bilinear'), reference_inputs_func=partial(reference_inputs_upsample, 'bilinear'), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('_upsample_bilinear2d_aa', op=torch.ops.aten._upsample_bilinear2d_aa, aten_name='_upsample_bilinear2d_aa', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.uint8), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=partial(sample_inputs_upsample_aa, 'bilinear'), supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestDTensorOps', 'test_dtensor_op_db'), DecorateInfo(unittest.expectedFailure, 'TestEagerFusionOpInfo', 'test_aot_autograd_symbolic_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestInductorOpInfo', 'test_comprehensive'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'))), OpInfo('nn.functional.soft_margin_loss', dtypes=floating_types_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, sample_inputs_func=partial(sample_inputs_loss, rhs_requires_grad=False), error_inputs_func=error_inputs_soft_margin_loss), OpInfo('nn.functional.upsample_nearest', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.uint8, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.uint8, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=partial(sample_inputs_upsample, 'nearest'), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), supports_out=False), OpInfo('nn.functional.margin_ranking_loss', dtypes=all_types_and(torch.half, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_margin_ranking_loss, error_inputs_func=error_inputs_margin_ranking_loss, reference_inputs_func=reference_inputs_margin_ranking_loss, supports_forward_ad=True, supports_fwgrad_bwgrad=True), OpInfo('nn.functional.multi_margin_loss', dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16), supports_out=False, supports_gradgrad=False, sample_inputs_func=sample_inputs_multi_margin_loss, reference_inputs_func=reference_inputs_multi_margin_loss, error_inputs_func=error_inputs_multi_margin_loss, decorators=(DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=0.0001)}), 'TestJit', 'test_variant_consistency_jit'),)), OpInfo('nn.functional.multilabel_margin_loss', dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16), supports_out=False, supports_gradgrad=False, sample_inputs_func=sample_inputs_multilabel_margin_loss, reference_inputs_func=reference_inputs_multilabel_margin_loss, error_inputs_func=error_inputs_multilabel_margin_loss), OpInfo('nn.functional.leaky_relu', aliases=None, aten_name='leaky_relu', aten_backward_name='leaky_relu_backward', sample_inputs_func=sample_inputs_leaky_relu, dtypes=floating_types_and(torch.bfloat16, torch.float16), inplace_variant=lambda x, negative_slope=0.01: torch.nn.functional.leaky_relu(x, negative_slope, inplace=True), supports_autograd=True, assert_autodiffed=True, supports_gradgrad=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_nonfusible_nodes=['aten::leaky_relu']), OpInfo('nn.functional.multilabel_soft_margin_loss', supports_out=False, dtypes=floating_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_multilabel_soft_margin_loss, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.0001, rtol=0.0001)}), 'TestJit', 'test_variant_consistency_jit'),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='cuda'),)), OpInfo('nn.functional.avg_pool2d', aten_name='avg_pool2d', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.int64, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), error_inputs_func=error_inputs_avg_pool2d, sample_inputs_func=sample_inputs_avgpool2d, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cuda'),)), OpInfo('nn.functional.fractional_max_pool2d', supports_autograd=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.fractional_max_pool2d, input, *args, **kwargs), check_batched_forward_grad=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), test_neg_view=False, sample_inputs_func=sample_inputs_fractional_max_pool2d, decorators=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), skips=(DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'),)), OpInfo('nn.functional.fractional_max_pool3d', supports_autograd=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.fractional_max_pool3d, input, *args, **kwargs), check_batched_forward_grad=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), test_neg_view=False, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, sample_inputs_func=sample_inputs_fractional_max_pool3d, decorators=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), skips=(DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'),)), OpInfo('nn.functional.max_pool1d', aten_name='max_pool1d', supports_autograd=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, assert_jit_shape_analysis=False, dtypes=floating_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.skip('Works on some configs'), 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestTags', 'test_tags')), error_inputs_func=error_inputs_max_pool1d, sample_inputs_func=sample_inputs_max_pool), OpInfo('nn.functional.max_pool2d', aten_name='max_pool2d', gradcheck_fast_mode=True, check_batched_gradgrad=False, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, assert_jit_shape_analysis=True, dtypes=all_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), error_inputs_func=error_inputs_max_pool2d, sample_inputs_func=sample_inputs_max_pool), OpInfo('max_pool2d_with_indices_backward', op=max_pool2d_backward, aten_name=None, method_variant=None, inplace_variant=None, operator_variant=None, inplace_operator_variant=None, check_batched_gradgrad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, assert_jit_shape_analysis=False, dtypes=floating_types_and(torch.bfloat16, torch.float16), sample_inputs_func=sample_inputs_max_pool, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), OpInfo('nn.functional.max_pool3d', aten_name='max_pool3d', gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, assert_jit_shape_analysis=False, dtypes=all_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, error_inputs_func=error_inputs_max_pool3d, sample_inputs_func=sample_inputs_max_pool), OpInfo('nn.functional.max_unpool1d', aten_name='max_unpool1d', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, assert_jit_shape_analysis=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), sample_inputs_func=sample_inputs_max_unpool, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD', active_if=not IS_MACOS), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_forward_ad', device_type='cpu'))), OpInfo('nn.functional.max_unpool1d', variant_test_name='grad', aten_name='max_unpool1d', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, assert_jit_shape_analysis=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), sample_inputs_func=sample_inputs_max_unpool_grad), OpInfo('nn.functional.max_unpool2d', aten_name='max_unpool2d', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, assert_jit_shape_analysis=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), sample_inputs_func=sample_inputs_max_unpool, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD', active_if=not IS_MACOS), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_forward_ad'))), OpInfo('nn.functional.max_unpool2d', variant_test_name='grad', aten_name='max_unpool2d', gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_grad=False, supports_out=False, assert_jit_shape_analysis=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), sample_inputs_func=sample_inputs_max_unpool_grad), OpInfo('nn.functional.max_unpool3d', aten_name='max_unpool3d', gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, assert_jit_shape_analysis=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), sample_inputs_func=sample_inputs_max_unpool, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD', active_if=not IS_MACOS), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_gradgrad'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_grad'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_forward_ad'))), OpInfo('nn.functional.max_unpool3d', variant_test_name='grad', aten_name='max_unpool3d', supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, assert_jit_shape_analysis=False, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16), sample_inputs_func=sample_inputs_max_unpool_grad), OpInfo('nn.functional.linear', aten_name='linear', supports_autograd=True, supports_gradgrad=True, sample_inputs_func=sample_inputs_linear, dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfROCM=floating_and_complex_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, supports_expanded_weight=True, decorators=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),)), OpInfo('nn.functional.bilinear', aten_name='bilinear', supports_autograd=True, sample_inputs_func=sample_inputs_bilinear, dtypes=all_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, *([torch.bfloat16] if SM53OrLater or TEST_WITH_ROCM else [])), decorators=(DecorateInfo(toleranceOverride({torch.float16: tol(atol=5e-05, rtol=0.001)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cpu'),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes', device_type='cuda', active_if=not SM53OrLater), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bfloat16,))), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('nn.functional.glu', aten_name='glu', gradcheck_fast_mode=True, sample_inputs_func=sample_inputs_glu, dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), UnaryUfuncInfo('nn.functional.elu', aten_backward_name='elu_backward', ref=lambda x, alpha=1.0, inplace=False: np.maximum(0.0, x) + np.minimum(0.0, alpha * (np.exp(x) - 1)), dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, sample_kwargs=lambda device, dtype, input: ({'alpha': 0.8}, {'alpha': 0.8}), inplace_variant=lambda x, alpha=1.0: torch.nn.functional.elu(x, alpha, inplace=True), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.0012), torch.bfloat16: tol(atol=0.001, rtol=0.0012)}), 'TestUnaryUfuncs', device_type='cuda')]), UnaryUfuncInfo('nn.functional.prelu', aten_backward_name='_prelu_kernel_backward', ref=lambda x, weight: np.maximum(0.0, x) + np.minimum(0.0, x) * (weight if x.ndim == 1 else weight.reshape([weight.size if i == 1 else 1 for i in range(0, x.ndim)])), dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, sample_kwargs=sample_kwargs_prelu_scalar_weight, error_inputs_func=error_inputs_prelu, sample_inputs_func=sample_inputs_prelu, reference_inputs_func=reference_inputs_prelu, decorators=[DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')]), UnaryUfuncInfo('nn.functional.celu', ref=lambda x, alpha=1.0, inplace=False: np.maximum(0.0, x) + np.minimum(0.0, alpha * (np.exp(x / alpha) - 1)), dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, sample_kwargs=lambda device, dtype, input: ({'alpha': 0.8}, {'alpha': 0.8}), inplace_variant=lambda x, alpha=1.0: torch.nn.functional.celu(x, alpha, inplace=True), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.0012), torch.bfloat16: tol(atol=0.001, rtol=0.0012)}), 'TestUnaryUfuncs', device_type='cuda')]), UnaryUfuncInfo('nn.functional.rrelu', aten_backward_name='rrelu_with_noise_backward', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.rrelu, input, *args, **kwargs), inplace_variant=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.rrelu, input, *args, inplace=True, **kwargs), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), gradcheck_wrapper=wrapper_set_seed, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, sample_kwargs=lambda device, dtype, input: (dict(lower=0.0, upper=1.0, training=True), dict(lower=0.0, upper=1.0, training=True)), sample_inputs_func=partial(sample_inputs_elementwise_unary, op_kwargs=dict(lower=0.0, upper=1.0, training=True)), error_inputs_func=error_inputs_rrelu, decorators=(DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.0012), torch.bfloat16: tol(atol=0.001, rtol=0.0012)}), 'TestUnaryUfuncs', device_type='cuda'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_inplace_forward_mode_AD'), DecorateInfo(unittest.skip('Different noise'), 'TestUnaryUfuncs', 'test_batch_vs_slicing'), DecorateInfo(unittest.skip('Different noise'), 'TestUnaryUfuncs', 'test_contig_vs_every_other'), DecorateInfo(unittest.skip('Different noise'), 'TestUnaryUfuncs', 'test_non_contig_expand'), DecorateInfo(unittest.skip('Different noise'), 'TestUnaryUfuncs', 'test_contig_vs_transposed'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), UnaryUfuncInfo('nn.functional.selu', ref=lambda x, inplace=False: 1.0507009873554805 * (np.maximum(0.0, x) + np.minimum(0.0, 1.6732632423543772 * (np.exp(x) - 1))), dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, inplace_variant=lambda x: torch.nn.functional.selu(x, inplace=True), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.018), torch.bfloat16: tol(atol=0.01, rtol=0.018)}), 'TestUnaryUfuncs', device_type='cuda')]), OpInfo('torch._scaled_mm', sample_inputs_func=sample_inputs_scaled_mm, dtypes=empty_types(), dtypesIfCUDA=empty_types() + (torch.float8_e4m3fn,), supports_out=True, supports_forward_ad=False, supports_autograd=False, decorators=[skipCUDAIf(not SM90OrLater or TEST_WITH_ROCM, 'Requires CUDA SM >= 9.0')], skips=()), OpInfo('nn.functional.scaled_dot_product_attention', op=lambda *args, **kwargs: wrapper_set_seed(torch.nn.functional.scaled_dot_product_attention, *args, **kwargs), sample_inputs_func=sample_inputs_scaled_dot_product_attention, dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=False, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, decorators=[DecorateInfo(toleranceOverride({torch.float32: tol(atol=5e-05, rtol=5e-06)}), 'TestCommon')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_backward', device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes', device_type='cuda', active_if=_get_torch_cuda_version() >= (11, 6)), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples', dtypes=(torch.float32,)), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_fn_fwgrad_bwgrad', device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients', 'test_fn_gradgrad', device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestMeta', 'test_dispatch_meta_outplace', device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestMeta', 'test_dispatch_symbolic_meta_outplace', device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_backward', device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped'), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('output is non-deterministic (when dropout_p > 0)'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip('This is '), 'TestInductorOpInfo', 'test_comprehensive'), DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', device_type='cuda', dtypes=(torch.bfloat16,), active_if=not SM80OrLater))), OpInfo('torch.ops.aten._efficient_attention_forward', sample_inputs_func=sample_inputs_efficient_attention_forward, dtypes=empty_types(), dtypesIfCUDA=custom_types(torch.float16, torch.float32) if not SM80OrLater else custom_types(torch.float16, torch.float32, torch.bfloat16), supports_out=False, supports_autograd=True, supports_fwgrad_bwgrad=False, supports_forward_ad=False, check_batched_forward_grad=False, decorators=[skipCUDAIf(TEST_WITH_ROCM, "ROCm doesn't support efficient attention")], skips=(DecorateInfo(unittest.expectedFailure, 'TestFakeTensor', 'test_fake_autocast', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestFakeTensor', 'test_fake', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_operator', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_noncontiguous_samples', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_backward', device_type='cuda'))), UnaryUfuncInfo('nn.functional.silu', aten_backward_name='silu_backward', ref=lambda x, inplace=False: x / (1 + np.exp(-x)), dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_autograd=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, supports_out=False, inplace_variant=lambda x: torch.nn.functional.silu(x, inplace=True), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.001), torch.bfloat16: tol(atol=0.0001, rtol=0.0001)}), 'TestUnaryUfuncs', device_type='cuda')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', dtypes=(torch.cfloat,), device_type='cpu'),), autodiff_nonfusible_nodes=['aten::silu']), UnaryUfuncInfo('nn.functional.silu', variant_test_name='complex', ref=lambda x, inplace=False: x / (1 + np.exp(-x)), dtypes=complex_types(), dtypesIfCUDA=complex_types(), supports_forward_ad=False, supports_autograd=False, assert_autodiffed=False, supports_out=False, inplace_variant=lambda x: torch.nn.functional.silu(x, inplace=True), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.001), torch.bfloat16: tol(atol=0.0001, rtol=0.0001)}), 'TestUnaryUfuncs', device_type='cuda')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', dtypes=(torch.cfloat,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.complex64, torch.cdouble)), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=(torch.complex64,)), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.complex64,)))), UnaryUfuncInfo('nn.functional.hardsigmoid', aten_backward_name='hardsigmoid_backward', ref=reference_hardsigmoid, dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_autograd=True, assert_autodiffed=False, supports_gradgrad=False, supports_forward_ad=True, supports_out=False, inplace_variant=partial(torch.nn.functional.hardsigmoid, inplace=True), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.0001, rtol=0.001)}), 'TestUnaryUfuncs', device_type='cuda')], skips=[DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_inplace_gradgrad'), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', active_if=TEST_WITH_ROCM)]), UnaryUfuncInfo('nn.functional.logsigmoid', aten_name='log_sigmoid', aten_backward_name='log_sigmoid_backward', ref=reference_logsigmoid, dtypes=floating_types_and(torch.half, torch.bfloat16), supports_autograd=True, assert_autodiffed=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_gradgrad=True, decorators=[DecorateInfo(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.005}), 'TestUnaryUfuncs', 'test_reference_numerics_small'), DecorateInfo(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.005}), 'TestUnaryUfuncs', 'test_reference_numerics_large'), DecorateInfo(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.005}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal')], skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning', device_type='cpu'),)), UnaryUfuncInfo('nn.functional.mish', aten_backward_name='mish_backward', ref=lambda x: x * np.tanh(reference_softplus(x)), dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, inplace_variant=partial(torch.nn.functional.mish, inplace=True), decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.001)}), 'TestUnaryUfuncs')]), UnaryUfuncInfo('nn.functional.softsign', ref=lambda x: x / (np.abs(x) + 1), dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.00013)}), 'TestUnaryUfuncs')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=(torch.int, torch.int8)),)), UnaryUfuncInfo('nn.functional.tanhshrink', ref=lambda x: x - np.tanh(x), dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, decorators=[DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.01, rtol=0.016)}), 'TestUnaryUfuncs'), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0006, rtol=1e-05), torch.bfloat16: tol(atol=0.01, rtol=0.016)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda')], skips=(DecorateInfo(unittest.skip('Fails on some jobs works on others!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.complex64, torch.complex128), active_if=IS_MACOS), DecorateInfo(unittest.skip('Fails on some jobs works on others!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.complex64, torch.complex128), device_type='cpu', active_if=IS_MACOS or IS_WINDOWS)), reference_numerics_filter=NumericsFilter(condition=lambda x: close_to_int(x / (math.pi * 0.5j)) if x.is_complex() else x.new_tensor(False, dtype=torch.bool), safe_val=0)), UnaryUfuncInfo('nn.functional.threshold', ref=lambda x, threshold, value: np.where(x <= threshold, value, x).astype(x.dtype), dtypes=all_types_and(torch.half, torch.bfloat16), inplace_variant=lambda x, threshold, value: torch.nn.functional.threshold(x, threshold, value, inplace=True), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=False, supports_gradgrad=True, supports_out=False, sample_kwargs=lambda device, dtype, input: ({'threshold': float.fromhex('0x1.3ap-3'), 'value': -9}, {'threshold': float.fromhex('0x1.3ap-3'), 'value': -9}), sample_inputs_func=sample_inputs_threshold), OpInfo('nn.functional.triplet_margin_loss', sample_inputs_func=sample_inputs_triplet_margin_loss, error_inputs_func=error_inputs_triplet_margin_loss, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True), OpInfo('nn.functional.triplet_margin_with_distance_loss', sample_inputs_func=partial(sample_inputs_triplet_margin_loss, with_distance=True), error_inputs_func=error_inputs_triplet_margin_loss, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'))), BinaryUfuncInfo('nextafter', dtypes=floating_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=floating_types_and(torch.bfloat16), supports_autograd=False, supports_rhs_python_scalar=False), OpInfo('to', op=lambda x, *args, **kwargs: x.to(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, sample_inputs_func=sample_inputs_to, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestMeta', 'test_meta_outplace'), DecorateInfo(unittest.skip('Skipped!'), 'TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'))), OpInfo('topk', dtypes=all_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, sample_inputs_func=sample_inputs_topk), OpInfo('nn.functional.batch_norm', aten_name='batch_norm', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, sample_inputs_func=sample_inputs_batch_norm, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', device_type='cpu', dtypes=(torch.bfloat16, torch.float16)), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_forward_ad', device_type='cuda', active_if=TEST_WITH_ROCM), DecorateInfo(toleranceOverride({torch.float32: tol(atol=5e-05, rtol=1e-05)}), 'TestCompositeCompliance', 'test_forward_ad', device_type='cpu'))), OpInfo('nn.functional.batch_norm', variant_test_name='without_cudnn', aten_name='batch_norm', dtypes=empty_types(), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=[onlyCUDA, disablecuDNN], skips=(DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.001, rtol=0.0001)}), 'TestJit', 'test_variant_consistency_jit'),), sample_inputs_func=sample_inputs_batch_norm), OpInfo('nn.functional.binary_cross_entropy', aten_backward_name='binary_cross_entropy_backward', sample_inputs_func=sample_inputs_binary_cross_entropy, dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, gradcheck_fast_mode=False, supports_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.001, rtol=0.001)}), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides')), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),)), BinaryUfuncInfo('igamma', dtypes=floating_types_and(torch.bfloat16, torch.float16), aliases=('torch.special.gammainc',), dtypesIfCUDA=floating_types(), supports_rhs_python_scalar=False, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'),)), BinaryUfuncInfo('igammac', dtypes=floating_types_and(torch.bfloat16, torch.float16), aliases=('torch.special.gammaincc',), dtypesIfCUDA=floating_types(), supports_autograd=False, supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'),)), UnaryUfuncInfo('nn.functional.softshrink', aten_name='softshrink', aten_backward_name='softshrink_backward', dtypes=floating_types_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=False, sample_inputs_func=sample_inputs_softshrink, error_inputs_func=error_inputs_softshrink), UnaryUfuncInfo('nn.functional.hardshrink', aten_name='hardshrink', aten_backward_name='hardshrink_backward', dtypes=floating_types_and(torch.bfloat16, torch.float16), assert_autodiffed=True, sample_inputs_func=sample_inputs_hardshrink, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_nonfusible_nodes=['aten::hardshrink']), UnaryUfuncInfo('nn.functional.hardtanh', aten_name='hardtanh', aten_backward_name='hardtanh_backward', dtypes=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64, torch.half, torch.bfloat16), backward_dtypes=all_types_and(torch.half, torch.bfloat16), backward_dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), assert_autodiffed=True, sample_inputs_func=sample_inputs_hardtanh, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_nonfusible_nodes=['aten::hardtanh']), OpInfo('nn.functional.gelu', aten_name='gelu', aten_backward_name='gelu_backward', ref=reference_gelu if TEST_SCIPY else None, error_inputs_func=error_inputs_gelu, supports_autograd=True, assert_autodiffed=True, sample_inputs_func=sample_inputs_gelu, dtypes=floating_types_and(torch.bfloat16, torch.half), supports_gradgrad=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_nonfusible_nodes=['aten::gelu'], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Unsupported on MPS for now'), 'TestCommon', 'test_numpy_ref_mps'))), UnaryUfuncInfo('nn.functional.relu6', aten_name='relu6', dtypes=all_types_and(torch.half, torch.bfloat16), backward_dtypes=floating_types_and(torch.half, torch.bfloat16), assert_autodiffed=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_nonfusible_nodes=['aten::relu6']), OpInfo('mm', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_mm, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)),)), OpInfo('mode', op=torch.mode, dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'),), sample_inputs_func=sample_inputs_mode), make_mvlgamma_opinfo(variant_test_name='mvlgamma_p_1', domain=(1, None), skips=skips_mvlgamma(), sample_kwargs=lambda device, dtype, input: ({'p': 1}, {'d': 1})), make_mvlgamma_opinfo(variant_test_name='mvlgamma_p_3', domain=(2, None), skips=skips_mvlgamma(), sample_kwargs=lambda device, dtype, input: ({'p': 3}, {'d': 3})), make_mvlgamma_opinfo(variant_test_name='mvlgamma_p_5', domain=(3, None), skips=skips_mvlgamma(), sample_kwargs=lambda device, dtype, input: ({'p': 5}, {'d': 5})), BinaryUfuncInfo('ne', ref=np.not_equal, aliases=('not_equal',), dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), always_returns_bool=True, supports_autograd=False, skips=()), OpInfo('narrow', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=partial(sample_inputs_narrow_narrow_copy, is_narrow=True), reference_inputs_func=partial(reference_inputs_narrow_narrow_copy, is_narrow=True), error_inputs_func=partial(error_inputs_narrow_narrow_copy, is_narrow=True, is_ref=False), skips=(DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_backward'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_forward_ad'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'))), OpInfo('narrow_copy', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_out=True, supports_forward_ad=False, supports_fwgrad_bwgrad=False, supports_autograd=False, sample_inputs_func=partial(sample_inputs_narrow_narrow_copy, is_narrow=False), reference_inputs_func=partial(reference_inputs_narrow_narrow_copy, is_narrow=False), error_inputs_func=partial(error_inputs_narrow_narrow_copy, is_narrow=False, is_ref=False), skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestLazyOpInfo', 'test_correctness'), DecorateInfo(unittest.expectedFailure, 'TestLazyOpInfo', 'test_correctness_with_reusing_ir'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_outplace', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), OpInfo('view_copy', dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16), ref=lambda x, newshape: np.reshape(x, newshape).copy(), supports_out=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_autograd=True, sample_inputs_func=sample_inputs_view_reshape, error_inputs_func=error_inputs_view_reshape), UnaryUfuncInfo('neg', aliases=('negative',), ref=np.negative, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.chalf), error_inputs_func=error_inputs_neg, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, assert_autodiffed=True), OpInfo('dist', op=torch.dist, dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_dist), OpInfo('outer', op=torch.outer, aliases=('ger',), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_outer), OpInfo('ormqr', op=torch.ormqr, dtypes=floating_and_complex_types(), gradcheck_fast_mode=True, supports_forward_ad=False, supports_fwgrad_bwgrad=False, sample_inputs_func=sample_inputs_ormqr, error_inputs_func=error_inputs_ormqr, decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack], skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),)), OpInfo('permute', ref=np.transpose, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, assert_autodiffed=True, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], assert_jit_shape_analysis=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_varargs=True, sample_inputs_func=sample_inputs_permute, reference_inputs_func=reference_inputs_permute), BinaryUfuncInfo('pow', dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.half, torch.bfloat16, torch.chalf), ref=np.power, backward_dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.bfloat16, torch.half, torch.chalf), gradcheck_fast_mode=True, supports_inplace_autograd=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, supports_one_python_scalar=True, rhs_make_tensor_kwargs=dict(low=0), lhs_make_tensor_kwargs=dict(low=0), decorators=(DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=1.3e-05)}), 'TestBinaryUfuncs', 'test_reference_numerics'), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=1.3e-05), torch.complex128: tol(atol=0.0001, rtol=1.3e-05)}), 'TestBinaryUfuncs', 'test_scalar_support')), skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=[torch.int8, torch.int16, torch.int32, torch.int64]), DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_reference_numerics_large_values', dtypes=[torch.int16, torch.int32, torch.int64]), DecorateInfo(unittest.skip('Skipped!'), 'TestDecomp', 'test_quick', dtypes=(torch.complex32,), active_if=TEST_WITH_ROCM), DecorateInfo(unittest.skip('Skipped!'), 'TestDecomp', 'test_comprehensive', dtypes=(torch.complex32,), active_if=TEST_WITH_ROCM), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_complex_half_reference_testing', dtypes=(torch.complex32,), active_if=TEST_WITH_ROCM), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_batch_vs_slicing', dtypes=(torch.complex32,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_non_contig', dtypes=(torch.complex32,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.complex32,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.complex32, torch.complex64, torch.complex128)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_large_values', dtypes=(torch.complex32, torch.complex64, torch.complex128)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=(torch.complex32, torch.complex64, torch.complex128)))), BinaryUfuncInfo('float_power', ref=np.float_power, dtypes=all_types_and_complex_and(torch.half, torch.bfloat16, torch.bool), promotes_int_to_float=True, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_one_python_scalar=True, rhs_make_tensor_kwargs=dict(low=0), lhs_make_tensor_kwargs=dict(low=0), decorators=(DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=1.3e-05), torch.complex128: tol(atol=0.0001, rtol=1.3e-05)}), 'TestBinaryUfuncs', 'test_scalar_support'),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=[torch.complex64, torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_large_values', dtypes=[torch.complex64, torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=[torch.complex64, torch.complex128]), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace', dtypes=[torch.bfloat16, torch.float16, torch.float32]))), OpInfo('qr', op=torch.qr, dtypes=floating_and_complex_types(), sample_inputs_func=sample_inputs_linalg_qr_geqrf, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_gradgrad=False, decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack]), UnaryUfuncInfo('rad2deg', ref=np.degrees, decorators=(precisionOverride({torch.bfloat16: 0.7, torch.float16: 0.7}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True), UnaryUfuncInfo('real', ref=np.real, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_out_arg_all_dtypes'),)), OpInfo('roll', ref=np.roll, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), error_inputs_func=error_inputs_roll, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_roll, decorators=(onlyNativeDeviceTypes,)), OpInfo('rot90', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half), error_inputs_func=error_inputs_rot90, gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_rot90), UnaryUfuncInfo('round', ref=np.round, aliases=('special.round',), dtypes=all_types_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=tuple((t for t in integral_types() if t != torch.uint8))), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bfloat16,))), supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, assert_autodiffed=True), UnaryUfuncInfo('round', ref=np.round, variant_test_name='decimals_0', aliases=('special.round',), dtypes=floating_types_and(torch.half, torch.bfloat16), sample_kwargs=lambda device, dtype, input: ({'decimals': 0}, {'decimals': 0}), sample_inputs_func=partial(sample_inputs_elementwise_unary, op_kwargs={'decimals': 0}), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=False, supports_sparse_csr=False), UnaryUfuncInfo('round', ref=np.round, variant_test_name='decimals_3', aliases=('special.round',), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), sample_kwargs=lambda device, dtype, input: ({'decimals': 3}, {'decimals': 3}), sample_inputs_func=partial(sample_inputs_elementwise_unary, op_kwargs={'decimals': 3}), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits'), DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.001, rtol=0.016)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda'), DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.001, rtol=0.016)}), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cuda')), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=False, supports_sparse_csr=False), UnaryUfuncInfo('round', ref=np.round, variant_test_name='decimals_neg_3', aliases=('special.round',), dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), sample_kwargs=lambda device, dtype, input: ({'decimals': -3}, {'decimals': -3}), sample_inputs_func=partial(sample_inputs_elementwise_unary, op_kwargs={'decimals': -3}), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits')), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=False, supports_sparse_csr=False), UnaryUfuncInfo('sin', ref=np.sin, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, handles_large_floats=False, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cdouble,), device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.cfloat, torch.cdouble), device_type='cpu', active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cfloat, torch.cdouble), device_type='cpu', active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad')), decorators=(precisionOverride({torch.bfloat16: 0.01}),)), UnaryUfuncInfo('sinc', ref=np_sinc_with_fp16_as_fp32, aliases=('special.sinc',), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), handles_large_floats=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.bfloat16: 0.01, torch.float16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=[torch.cfloat]),)), UnaryUfuncInfo('sinh', ref=np_unary_ufunc_integer_promotion_wrapper(np.sinh), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.float16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cdouble,)), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.int8]), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'))), UnaryUfuncInfo('sign', ref=reference_sign, dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half), dtypesIfCUDA=all_types_and(torch.bool, torch.bfloat16, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),)), UnaryUfuncInfo('sgn', ref=reference_sgn, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), backward_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.half), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.bfloat16, torch.half, torch.chalf), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'))), OpInfo('split', dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool, torch.chalf), sample_inputs_func=partial(sample_inputs_split, list_args=False), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], assert_autodiffed=True), OpInfo('split', decomp_aten_name='split_with_sizes', variant_test_name='list_args', dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool), sample_inputs_func=partial(sample_inputs_split, list_args=True), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('unsafe_split', dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool, torch.chalf), sample_inputs_func=partial(sample_inputs_split, list_args=False), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], assert_autodiffed=True, check_batched_forward_grad=False), OpInfo('split_with_sizes', dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool, torch.chalf), sample_inputs_func=sample_inputs_split_with_sizes, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True), BinaryUfuncInfo('__radd__', op=torch.Tensor.__radd__, dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool), supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_nonfusible_nodes=['aten::add']), BinaryUfuncInfo('__rdiv__', op=torch.Tensor.__rdiv__, dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool), promotes_int_to_float=True, lhs_make_tensor_kwargs={'exclude_zero': True}, gradcheck_fast_mode=True, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, autodiff_nonfusible_nodes=['aten::mul', 'aten::reciprocal']), BinaryUfuncInfo('__rmul__', op=torch.Tensor.__rmul__, dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool), supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, autodiff_nonfusible_nodes=['aten::mul']), BinaryUfuncInfo('__rand__', op=torch.Tensor.__rand__, dtypes=integral_types_and(torch.bool), supports_out=False, supports_autograd=False, supports_forward_ad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), BinaryUfuncInfo('__ror__', op=torch.Tensor.__ror__, dtypes=integral_types_and(torch.bool), supports_out=False, supports_autograd=False, supports_forward_ad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), BinaryUfuncInfo('__rxor__', op=torch.Tensor.__rxor__, dtypes=integral_types_and(torch.bool), supports_out=False, supports_autograd=False, supports_forward_ad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), OpInfo('__rmatmul__', op=torch.Tensor.__rmatmul__, dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, *([torch.bfloat16] if SM53OrLater or TEST_WITH_ROCM else [])), assert_autodiffed=True, sample_inputs_func=partial(sample_inputs_matmul, is_rmatmul=True), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, decorators=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes', device_type='cuda', active_if=not SM53OrLater), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=0.0012)}), 'TestMathBits', 'test_conj_view'), DecorateInfo(toleranceOverride({torch.float32: tol(atol=1e-05, rtol=0.0012)}), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=1e-05, rtol=1e-05)}), 'TestDecomp', 'test_comprehensive', device_type='cuda', active_if=TEST_WITH_ROCM)), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('67470!'), 'TestCommon', 'test_noncontiguous_samples', device_type='cpu', dtypes=(torch.long,)), DecorateInfo(unittest.skip('Skipped!'), 'TestOpInfo', device_type='xla', dtypes=(torch.long,)), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness', device_type='cpu', dtypes=(torch.long,)))), BinaryUfuncInfo('__rmod__', op=torch.Tensor.__rmod__, dtypes=floating_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.half), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), assert_autodiffed=True, autodiff_nonfusible_nodes=['aten::remainder']), BinaryUfuncInfo('__rpow__', op=torch.Tensor.__rpow__, dtypes=all_types_and_complex_and(torch.bfloat16, torch.half), backward_dtypes=all_types_and_complex_and(torch.bfloat16, torch.half), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients')), assert_autodiffed=True, autodiff_nonfusible_nodes=['aten::pow']), BinaryUfuncInfo('__rsub__', op=torch.Tensor.__rsub__, dtypes=all_types_and_complex_and(torch.bfloat16, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), assert_autodiffed=True, autodiff_nonfusible_nodes=['aten::rsub']), BinaryUfuncInfo('rsub', dtypes=all_types_and_complex_and(torch.bfloat16, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, supports_inplace_autograd=False, assert_autodiffed=None, sample_inputs_func=sample_inputs_add_sub), OpInfo('select', aten_backward_name='select_backward', dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool, torch.chalf), sample_inputs_func=sample_inputs_select, assert_jit_shape_analysis=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('select_scatter', dtypes=all_types_and(torch.bfloat16, torch.half, torch.bool), sample_inputs_func=sample_inputs_select_scatter, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), OpInfo('slice', op=torch.ops.aten.slice.Tensor, dtypes=all_types_and_complex_and(torch.bfloat16, torch.half, torch.bool, torch.chalf), sample_inputs_func=sample_inputs_slice, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_scripting=False, supports_inplace_autograd=False, supports_out=False), OpInfo('slice_scatter', dtypes=all_types_and(torch.bfloat16, torch.half, torch.bool), sample_inputs_func=sample_inputs_slice_scatter, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False), UnaryUfuncInfo('signbit', ref=np.signbit, dtypes=all_types_and(torch.bool, torch.bfloat16, torch.half), supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_autograd=False), UnaryUfuncInfo('tan', ref=np.tan, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), decorators=(DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=1e-05)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda'),), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad')), reference_numerics_filter=NumericsFilter(condition=lambda x: close_to_int(x / (math.pi * 0.5)), safe_val=math.pi)), UnaryUfuncInfo('tanh', ref=np.tanh, aten_backward_name='tanh_backward', aliases=('nn.functional.tanh',), decorators=(precisionOverride({torch.bfloat16: 0.01}), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=2e-05)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda')), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, assert_jit_shape_analysis=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad')), reference_numerics_filter=NumericsFilter(condition=lambda x: close_to_int(x / (math.pi * 0.5j)) if x.is_complex() else x.new_tensor(False, dtype=torch.bool), safe_val=0)), OpInfo('tensor_split', ref=np.array_split, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_backward'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_forward_ad')), sample_inputs_func=sample_inputs_tensor_split), OpInfo('hsplit', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.bfloat16, torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_hsplit, error_inputs_func=error_inputs_hsplit), OpInfo('vsplit', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.bfloat16, torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_vsplit, error_inputs_func=error_inputs_vsplit), OpInfo('dsplit', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.bfloat16, torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_dsplit, error_inputs_func=error_inputs_dsplit), OpInfo('triangular_solve', op=torch.triangular_solve, dtypes=floating_and_complex_types(), sample_inputs_func=sample_inputs_legacy_solve, check_batched_gradgrad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_wrapper=lambda *args, **kwargs: gradcheck_wrapper_triangular_input(*args, idx=1, **kwargs), decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack], skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_fn_fwgrad_bwgrad', dtypes=floating_and_complex_types()), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='mps', dtypes=[torch.float32]))), UnaryUfuncInfo('trunc', aliases=('fix',), ref=np.trunc, dtypes=all_types_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=tuple((t for t in integral_types() if t != torch.uint8))),), supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, assert_autodiffed=True), UnaryUfuncInfo('exp2', aliases=('special.exp2',), ref=np_unary_ufunc_integer_promotion_wrapper(np.exp2), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]))), UnaryUfuncInfo('expm1', aliases=('special.expm1',), ref=np_unary_ufunc_integer_promotion_wrapper(np.expm1), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, promotes_int_to_float=True, assert_autodiffed=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.complex128]), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'))), UnaryUfuncInfo('nan_to_num', ref=np.nan_to_num, dtypes=all_types_and(torch.half, torch.bool, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.half, torch.bool, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse=True, skips=(DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),), sample_kwargs=lambda device, dtype, input: ({}, {'posinf': torch.finfo(torch.bfloat16).max, 'neginf': torch.finfo(torch.bfloat16).min}) if dtype is torch.bfloat16 else ({}, {})), UnaryUfuncInfo('reciprocal', ref=np_unary_ufunc_integer_promotion_wrapper(np.reciprocal), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble]),)), UnaryUfuncInfo('rsqrt', ref=lambda x: np.reciprocal(np.sqrt(x)), domain=(0, None), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), decorators=(precisionOverride({torch.half: 0.05}),), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.cfloat, torch.cdouble)), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.chalf,)))), UnaryUfuncInfo('sqrt', ref=np.sqrt, supports_sparse=True, domain=(0, None), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.bfloat16: 0.07}), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.01, rtol=0)}), 'TestUnaryUfuncs', 'test_reference_numerics_large')), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=(torch.cfloat, torch.cdouble), active_if=IS_MACOS), DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'))), UnaryUfuncInfo('square', ref=np.square, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), decorators=(precisionOverride({torch.complex64: 0.0003, torch.bfloat16: 0.3}),), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_meta_inplace', dtypes=[torch.bool]), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_inplace', dtypes=[torch.bool]), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_inplace', dtypes=[torch.bool]))), OpInfo('lerp', dtypes=floating_and_complex_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=floating_and_complex_types_and(torch.chalf, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_lerp, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True), UnaryUfuncInfo('angle', ref=np.angle, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool), decorators=(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.01}),), backward_dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.chalf), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_complex_to_float=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=(torch.bfloat16, torch.float16, torch.float32, torch.float64)),)), UnaryUfuncInfo('isfinite', ref=np.isfinite, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_out=False, supports_autograd=False), UnaryUfuncInfo('isinf', ref=np.isinf, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_out=False, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_autograd=False), UnaryUfuncInfo('isposinf', ref=np.isposinf, dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16), supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_autograd=False), UnaryUfuncInfo('isneginf', ref=np.isneginf, dtypes=all_types_and(torch.bool, torch.bfloat16, torch.float16), supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_autograd=False), UnaryUfuncInfo('isreal', ref=np.isreal, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), supports_out=False, supports_autograd=False), UnaryUfuncInfo('isnan', ref=np.isnan, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), supports_out=False, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_autograd=False), OpInfo('einsum', op=lambda tensors, equation: torch.einsum(equation, tensors), dtypes=all_types_and_complex_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.half, *([torch.bfloat16] if SM60OrLater or TEST_WITH_ROCM else [])), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_einsum, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'))), OpInfo('svd', op=torch.svd, dtypes=floating_and_complex_types(), sample_inputs_func=sample_inputs_svd, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, check_batched_grad=False, check_batched_gradgrad=False, decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack, with_tf32_off], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestFakeTensor', 'test_fake_crossref_backward_amp', device_type='cuda', dtypes=[torch.float32], active_if=TEST_WITH_ROCM), DecorateInfo(unittest.skip('Skipped!'), 'TestFakeTensor', 'test_fake_crossref_backward_no_amp', device_type='cuda', dtypes=[torch.float32], active_if=TEST_WITH_ROCM))), OpInfo('svd_lowrank', op=lambda *args, **kwargs: wrapper_set_seed(lambda a, b, **kwargs: torch.svd_lowrank(a @ b.mT, **kwargs), *args, **kwargs), dtypes=floating_types(), gradcheck_fast_mode=True, supports_out=False, check_batched_grad=False, check_batched_gradgrad=False, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, supports_forward_ad=True, sample_inputs_func=sample_inputs_svd_lowrank, decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack, with_tf32_off, DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.001, rtol=0.001)}), 'TestCommon', 'test_noncontiguous_samples', device_type='cuda')], skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(slowTest, 'TestCompositeCompliance', 'test_forward_ad'))), OpInfo('pca_lowrank', op=lambda *args, **kwargs: wrapper_set_seed(lambda a, b, **kwargs: torch.pca_lowrank(a @ b.mT, **kwargs), *args, **kwargs), dtypes=floating_types(), gradcheck_fast_mode=True, supports_out=False, check_batched_forward_grad=False, check_batched_grad=False, check_batched_gradgrad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_pca_lowrank, decorators=[skipCUDAIfNoCusolver, skipCPUIfNoLapack, with_tf32_off, DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.001, rtol=0.001)}), 'TestCommon', 'test_noncontiguous_samples', device_type='cuda')], skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), BinaryUfuncInfo('polar', dtypes=floating_types(), supports_forward_ad=True, lhs_make_tensor_kwargs=dict(low=0), supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_fn_fwgrad_bwgrad'))), UnaryUfuncInfo('polygamma', op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs), variant_test_name='polygamma_n_0', ref=reference_polygamma if TEST_SCIPY else None, dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, sample_inputs_func=sample_inputs_polygamma, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),), sample_kwargs=lambda device, dtype, input: ({'n': 0}, {'n': 0})), UnaryUfuncInfo('polygamma', op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs), variant_test_name='polygamma_n_1', ref=reference_polygamma if TEST_SCIPY else None, dtypes=all_types_and(torch.bool, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, sample_inputs_func=sample_inputs_polygamma, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large')), sample_kwargs=lambda device, dtype, input: ({'n': 1}, {'n': 1}), reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)), UnaryUfuncInfo('polygamma', op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs), variant_test_name='polygamma_n_2', ref=reference_polygamma if TEST_SCIPY else None, dtypes=all_types_and(torch.bool, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, sample_inputs_func=sample_inputs_polygamma, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal')), sample_kwargs=lambda device, dtype, input: ({'n': 2}, {'n': 2}), reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)), UnaryUfuncInfo('polygamma', op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs), variant_test_name='polygamma_n_3', ref=reference_polygamma if TEST_SCIPY else None, dtypes=all_types_and(torch.bool, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, sample_inputs_func=sample_inputs_polygamma, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal')), sample_kwargs=lambda device, dtype, input: ({'n': 3}, {'n': 3}), reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)), UnaryUfuncInfo('polygamma', op=lambda x, n, **kwargs: torch.polygamma(n, x, **kwargs), variant_test_name='polygamma_n_4', ref=reference_polygamma if TEST_SCIPY else None, decorators=(precisionOverride({torch.float16: 0.0005, torch.float32: 0.0005}),), dtypes=all_types_and(torch.bool, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, sample_inputs_func=sample_inputs_polygamma, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal')), sample_kwargs=lambda device, dtype, input: ({'n': 4}, {'n': 4}), reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)), OpInfo('ravel', ref=np.ravel, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_ravel), OpInfo('unravel_index', ref=np.unravel_index, dtypes=integral_types_and(), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_unravel_index), OpInfo('reshape', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_view_reshape, reference_inputs_func=reference_inputs_view_reshape, error_inputs_func=error_inputs_view_reshape, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True), OpInfo('reshape_as', op=lambda x, other: x.reshape_as(other), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=partial(sample_inputs_view_reshape, tensor_arg=True), reference_inputs_func=partial(reference_inputs_view_reshape, tensor_arg=True), error_inputs_func=partial(error_inputs_view_reshape, tensor_arg=True), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), OpInfo('view', op=lambda x, shape: x.view(shape), dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, sample_inputs_func=sample_inputs_view_reshape, reference_inputs_func=reference_inputs_view_reshape, error_inputs_func=error_inputs_view_reshape, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), OpInfo('view_as', op=lambda x, other: x.view_as(other), dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=partial(sample_inputs_view_reshape, tensor_arg=True), reference_inputs_func=partial(reference_inputs_view_reshape, tensor_arg=True), error_inputs_func=partial(error_inputs_view_reshape, tensor_arg=True), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), OpInfo('atleast_1d', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_atleast1d2d3d, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32]))), OpInfo('atleast_2d', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32])), sample_inputs_func=sample_inputs_atleast1d2d3d), OpInfo('atleast_3d', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=[torch.float32])), sample_inputs_func=sample_inputs_atleast1d2d3d), OpInfo('flatten', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), ref=reference_flatten, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_flatten, reference_inputs_func=reference_inputs_flatten), OpInfo('unflatten', op=torch.unflatten, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_unflatten), OpInfo('column_stack', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),), sample_inputs_func=sample_inputs_column_stack), OpInfo('pinverse', op=torch.pinverse, dtypes=floating_and_complex_types(), check_batched_grad=False, check_batched_gradgrad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, supports_out=False, sample_inputs_func=sample_inputs_linalg_invertible, decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager', device_type='mps', dtypes=[torch.float32]), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='mps', dtypes=[torch.float32]))), OpInfo('gather', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_gather, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, supports_forward_ad=True, supports_fwgrad_bwgrad=True, error_inputs_func=error_inputs_gather), OpInfo('index_fill', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.complex32), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestFakeTensor', 'test_fake_crossref_backward_no_amp'), DecorateInfo(unittest.expectedFailure, 'TestFakeTensor', 'test_fake_crossref_backward_amp')), sample_inputs_func=sample_inputs_index, reference_inputs_func=partial(sample_inputs_index, reference=True)), OpInfo('index_copy', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.complex32), supports_out=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_index, reference_inputs_func=partial(sample_inputs_index, reference=True), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL), OpInfo('index_select', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_index, reference_inputs_func=partial(sample_inputs_index, reference=True), error_inputs_func=error_inputs_index_select, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL), OpInfo('index_add', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_index, reference_inputs_func=partial(sample_inputs_index, reference=True), error_inputs_func=error_inputs_index_add, skips=(DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bool,)),), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL), OpInfo('index_reduce', dtypes=all_types_and(torch.float16, torch.bfloat16), supports_out=True, sample_inputs_func=sample_inputs_index_reduce), OpInfo('__getitem__', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_inplace_autograd=False, supports_scripting=False, op=torch.Tensor.__getitem__, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='cuda')), sample_inputs_func=sample_inputs_getitem), OpInfo('index_put', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_inplace_autograd=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, test_neg_view=False, sample_inputs_func=sample_inputs_index_put, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped'), 'TestBwdGradients', 'test_fn_grad', dtypes=[torch.float64], device_type='cuda', active_if=TEST_WITH_ROCM and TEST_WITH_TORCHINDUCTOR))), OpInfo('sort', dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_sort, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=()), OpInfo('unique', dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.float16), sample_inputs_func=sample_inputs_unique, supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Output order is undefined when sorted=False'), 'TestCommon', 'test_compare_cpu'))), OpInfo('unique_consecutive', dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.float16), sample_inputs_func=sample_inputs_unique_consecutive, supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), OpInfo('put', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, check_batched_gradgrad=False, sample_inputs_func=sample_inputs_put), OpInfo('take', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), check_batched_grad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_take, error_inputs_func=error_inputs_take), OpInfo('scatter', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_scatter, error_inputs_func=error_inputs_scatter_and_scatter_add), UnaryUfuncInfo('bfloat16', op=lambda x, *args, **kwargs: x.bfloat16(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, skips=(DecorateInfo(unittest.expectedFailure, 'TestFwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'))), UnaryUfuncInfo('bool', op=lambda x, *args, **kwargs: x.bool(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), UnaryUfuncInfo('byte', op=lambda x, *args, **kwargs: x.byte(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'))), UnaryUfuncInfo('char', op=lambda x, *args, **kwargs: x.char(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'))), UnaryUfuncInfo('double', op=lambda x, *args, **kwargs: x.double(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), UnaryUfuncInfo('float', op=lambda x, *args, **kwargs: x.float(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, skips=(DecorateInfo(unittest.expectedFailure, 'TestFwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), UnaryUfuncInfo('half', op=lambda x, *args, **kwargs: x.half(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_autograd=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestFwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), UnaryUfuncInfo('int', op=lambda x, *args, **kwargs: x.int(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'))), UnaryUfuncInfo('long', op=lambda x, *args, **kwargs: x.long(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'))), UnaryUfuncInfo('short', op=lambda x, *args, **kwargs: x.short(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'))), UnaryUfuncInfo('cdouble', op=torch.Tensor.cdouble, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'))), UnaryUfuncInfo('cfloat', op=torch.Tensor.cfloat, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, skips=(DecorateInfo(unittest.expectedFailure, 'TestFwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'))), UnaryUfuncInfo('chalf', op=lambda x, *args, **kwargs: x.chalf(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_conversion, skips=(DecorateInfo(unittest.expectedFailure, 'TestFwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager', device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view', device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view', device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'))), OpInfo('empty_like', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_like_fns, reference_inputs_func=reference_inputs_like_fns, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values'), DecorateInfo(unittest.skip('Expected: empty_like is not comparable'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('zeros_like', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_like_fns, supports_autograd=False, error_inputs_sparse_func=error_inputs_sparse_like_fns, sample_inputs_sparse_coo_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_coo), sample_inputs_sparse_csr_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_csr), sample_inputs_sparse_csc_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_csc), sample_inputs_sparse_bsr_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_bsr), sample_inputs_sparse_bsc_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_bsc), skips=()), OpInfo('ones_like', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_like_fns, supports_autograd=False, skips=()), OpInfo('randn', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16, torch.complex32), op=lambda *args, **kwargs: wrapper_set_seed(torch.randn, *args, **kwargs), supports_out=True, sample_inputs_func=sample_inputs_randn, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Test expects tensor input'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_vmap_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestDecomp', 'test_quick'))), OpInfo('randn_like', dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16, torch.complex32), op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.randn_like, inp, *args, **kwargs), supports_out=False, sample_inputs_func=sample_inputs_like_fns, supports_autograd=False, error_inputs_sparse_func=error_inputs_sparse_like_fns, sample_inputs_sparse_coo_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_coo), sample_inputs_sparse_csr_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_csr), sample_inputs_sparse_csc_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_csc), sample_inputs_sparse_bsr_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_bsr), sample_inputs_sparse_bsc_func=partial(sample_inputs_sparse_like_fns, layout=torch.sparse_bsc), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Expected: randn_like is not comparable between dtypes'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('rand_like', dtypes=floating_types_and(torch.half, torch.bfloat16, torch.complex32, torch.complex64, torch.complex128), op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.randn_like, inp, *args, **kwargs), supports_out=False, sample_inputs_func=sample_inputs_like_fns, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Expected: randn_like is not comparable between dtypes'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('randint', dtypes=all_types_and(torch.half, torch.bfloat16), op=lambda *args, **kwargs: wrapper_set_seed(torch.randint, *args, **kwargs), supports_out=False, sample_inputs_func=sample_inputs_randint, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Test expects tensor input'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_vmap_exhaustive'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_multiple_devices', dtypes=[torch.float32, torch.int64], active_if=TEST_WITH_ROCM))), OpInfo('randint_like', dtypes=all_types_and(torch.half, torch.bfloat16), op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.randint_like, inp, *args, **kwargs), supports_out=False, sample_inputs_func=sample_inputs_randint_like, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('full_like', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_full_like, supports_autograd=False, skips=()), OpInfo('new_zeros', op=lambda x, *args, **kwargs: x.new_zeros(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_new_fns, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),), supports_autograd=False), OpInfo('new_ones', op=lambda x, *args, **kwargs: x.new_ones(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_new_fns, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),), supports_autograd=False), OpInfo('ones', op=torch.ones, supports_autograd=False, supports_varargs=True, is_factory_function=True, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=True, sample_inputs_func=sample_inputs_ones_zeros, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'))), OpInfo('zeros', op=torch.zeros, supports_autograd=False, is_factory_function=True, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=True, sample_inputs_func=sample_inputs_ones_zeros, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'))), OpInfo('full', op=torch.full, supports_autograd=False, is_factory_function=True, dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=True, sample_inputs_func=sample_inputs_full, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness', dtypes=(torch.bool,)))), OpInfo('new_empty', op=lambda x, *args, **kwargs: x.new_empty(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_new_fns, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values'), DecorateInfo(unittest.skip('Expected: new_empty is not comparable'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Expected: new_empty is not comparable'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu')), supports_autograd=False), OpInfo('new_empty_strided', op=lambda x, *args, **kwargs: x.new_empty_strided(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=partial(sample_inputs_new_fns, is_strided=True), supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestLazyOpInfo', 'test_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestLazyOpInfo', 'test_correctness_with_reusing_ir'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestCommon', 'test_non_standard_bool_values'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestDecomp', 'test_quick'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestProxyTensorOpInfo', 'test_make_fx_exhaustive'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestProxyTensorOpInfo', 'test_make_fx_fake_exhaustive'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestProxyTensorOpInfo', 'test_make_fx_symbolic_exhaustive'), DecorateInfo(unittest.skip('Expected: new_empty_strided is not comparable'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('empty_strided', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.empty_strided, inp, *args, **kwargs), dtypes=all_types_and_complex_and(torch.bfloat16, torch.bool, torch.half), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_empty_strided, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestLazyOpInfo'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace'), DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), OpInfo('empty', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_empty, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestLazyOpInfo'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('eye', dtypes=all_types_and_complex_and(torch.bool, torch.half), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_eye, error_inputs_func=error_inputs_eye, supports_out=True, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'))), OpInfo('empty_permuted', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_empty_permuted, error_inputs_func=error_inputs_empty_permuted, supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values'), DecorateInfo(unittest.skip('Expected: empty_permuted is not comparable'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'), DecorateInfo(unittest.skip('Expected: empty_permuted is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: empty_permuted is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: empty_permuted is not comparable'), 'TestLazyOpInfo'), DecorateInfo(unittest.skip('Expected: empty_permuted is not comparable'), 'TestCommon', 'test_complex_half_reference_testing'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('scalar_tensor', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_scalar_tensor, supports_autograd=False, supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'))), OpInfo('new_full', op=lambda x, *args, **kwargs: x.new_full(*args, **kwargs), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_out=False, sample_inputs_func=sample_inputs_new_full, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),), supports_autograd=False), OpInfo('multinomial', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.multinomial, inp, *args, **kwargs), method_variant=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.multinomial, inp, *args, **kwargs), dtypes=floating_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=floating_types_and(torch.half), supports_out=True, sample_inputs_func=sample_inputs_multinomial, error_inputs_func=error_inputs_multinomial, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu')), supports_autograd=False), OpInfo('normal', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.normal, inp, *args, **kwargs), inplace_variant=None, dtypes=floating_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.half), supports_out=True, sample_inputs_func=sample_inputs_normal_tensor_first, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Gradients are incorrect!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Gradients are incorrect!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('Skipped!'), 'TestDecomp', 'test_quick'), DecorateInfo(unittest.skip('Skipped!'), 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'))), OpInfo('normal', variant_test_name='number_mean', op=lambda std, mean, *args, **kwargs: wrapper_set_seed(torch.normal, mean, std, *args, **kwargs), inplace_variant=None, dtypes=floating_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.half), supports_out=True, sample_inputs_func=sample_inputs_normal_tensor_second, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_backward'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestBwdGradients'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestEagerFusionOpInfo'), DecorateInfo(unittest.skip('Skipped!'), 'TestOperators'), DecorateInfo(unittest.skip('Skipped!'), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('Skipped!'), 'TestDecomp', 'test_quick'), DecorateInfo(unittest.skip('Skipped!'), 'TestFakeTensor', device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestDeviceUtils', 'test_device_mode_ops'))), OpInfo('bernoulli', op=lambda inp, *args, **kwargs: wrapper_set_seed(torch.bernoulli, inp, *args, **kwargs), inplace_variant=None, method_variant=lambda inp, *args, **kwargs: wrapper_set_seed(torch.Tensor.bernoulli, inp, *args, **kwargs), dtypes=floating_types_and(torch.bfloat16, torch.half), supports_out=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_bernoulli, error_inputs_func=error_inputs_bernoulli, skips=(DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), OpInfo('scatter_add', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_scatter_add, error_inputs_func=error_inputs_scatter_and_scatter_add, supports_forward_ad=True, supports_fwgrad_bwgrad=True), OpInfo('stack', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_stack, assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'))), OpInfo('hstack', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_hstack_dstack_vstack, error_inputs_func=error_inputs_hstack_dstack_vstack, supports_forward_ad=True, supports_fwgrad_bwgrad=True), BinaryUfuncInfo('hypot', dtypes=floating_types_and(torch.bfloat16, torch.half), dtypesIfCUDA=floating_types_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_rhs_python_scalar=False), OpInfo('histogram', dtypes=floating_types(), dtypesIfCUDA=_dispatch_dtypes(), sample_inputs_func=sample_inputs_histogram, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestOpInfo', device_type='xla'))), OpInfo('histogramdd', dtypes=floating_types(), dtypesIfCUDA=_dispatch_dtypes(), sample_inputs_func=sample_inputs_histogramdd, error_inputs_func=error_inputs_histogramdd, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_errors', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), OpInfo('histc', dtypes=floating_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.int8, torch.int16, torch.int32, torch.int64), sample_inputs_func=sample_inputs_histc, supports_out=True, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out', device_type='cuda'),)), OpInfo('bincount', dtypes=integral_types_and(), sample_inputs_func=sample_inputs_bincount, supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'),)), OpInfo('bucketize', dtypes=all_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16), sample_inputs_func=sample_inputs_bucketize, reference_inputs_func=reference_inputs_bucketize, error_inputs_func=error_inputs_bucketize, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Expected failure!'), 'TestJit', 'test_variant_consistency_jit'),)), OpInfo('searchsorted', dtypes=all_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and(torch.bfloat16, torch.float16), sample_inputs_func=sample_inputs_searchsorted, supports_autograd=False, ref=reference_searchsorted, skips=(DecorateInfo(unittest.skip('Expected failure!'), 'TestJit', 'test_variant_consistency_jit'),)), OpInfo('cat', ref=_cat_np, aliases=('concat', 'concatenate'), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.complex32), sample_inputs_func=sample_inputs_cat_concat, reference_inputs_func=reference_inputs_cat, error_inputs_func=error_inputs_cat, gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, assert_autodiffed=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_numpy_ref_mps'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'), DecorateInfo(unittest.expectedFailure, 'TestNNCOpInfo', 'test_nnc_correctness'), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_gradgrad'))), OpInfo('unbind', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), ref=reference_unbind, sample_inputs_func=sample_inputs_unbind, error_inputs_func=error_inputs_unbind, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_gradgrad=True, supports_out=False), OpInfo('vstack', aliases=('row_stack',), dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_hstack_dstack_vstack, error_inputs_func=error_inputs_hstack_dstack_vstack, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_jit_alias_remapping'),)), OpInfo('dstack', dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_hstack_dstack_vstack, error_inputs_func=error_inputs_hstack_dstack_vstack, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False), OpInfo('unfold', op=lambda x, *args: x.unfold(*args), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), backward_dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_gradgrad=False, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive')), sample_inputs_func=sample_inputs_unfold), OpInfo('unfold_copy', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), backward_dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_gradgrad=False, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_unfold), OpInfo('msort', dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), check_batched_gradgrad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_msort, skips=()), OpInfo('movedim', aliases=('moveaxis',), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_movedim_moveaxis, reference_inputs_func=reference_movedim_moveaxis, error_inputs_func=error_movedim_moveaxis), OpInfo('renorm', dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_renorm, error_inputs_func=error_inputs_renorm, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Inconsistent accuracy'), 'TestDecomp', 'test_comprehensive', device_type='cpu', dtypes=(torch.float16,)),)), ShapeFuncInfo('repeat', op=lambda x, dims: x.repeat(dims), ref=np.tile, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_repeat_tile, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'),)), OpInfo('squeeze', ref=_squeeze_ref, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, assert_autodiffed=True, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], assert_jit_shape_analysis=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_inplace_batched_forward_grad=False, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_squeeze), OpInfo('squeeze', ref=_squeeze_ref, variant_test_name='multiple', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, assert_autodiffed=True, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_inplace_batched_forward_grad=False, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_squeeze_multiple), UnaryUfuncInfo('fill', ref=_fill_np, method_variant=None, sample_kwargs=_fill_sample_kwargs, sample_inputs_func=partial(sample_inputs_elementwise_unary, op_kwargs={'value': True}), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, dtypes=all_types_and_complex_and(torch.complex32, torch.bool, torch.float16, torch.bfloat16), supports_out=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('No fill_ op'), 'TestCudaFuserOpInfo'), DecorateInfo(unittest.skip('No fill_ op'), 'TestNNCOpInfo'))), OpInfo('resize_', op=lambda x, shape: x.clone().resize_(shape), method_variant=None, inplace_variant=torch.Tensor.resize_, test_neg_view=False, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Allowed exception'), 'TestCompositeCompliance', 'test_operator')), sample_inputs_func=sample_inputs_resize_ops), OpInfo('resize_as_', op=lambda x, other: torch.resize_as_(x.clone(), other), method_variant=None, inplace_variant=torch.Tensor.resize_as_, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.skip('Allowed exemption'), 'TestCompositeCompliance', 'test_operator')), sample_inputs_func=sample_inputs_resize_ops), OpInfo('take_along_dim', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_inplace_autograd=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_take_along_dim, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, decorators=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'),)), ShapeFuncInfo('tile', ref=np.tile, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_repeat_tile), OpInfo('trapz', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_trapezoid), OpInfo('trapezoid', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_trapezoid), OpInfo('cumulative_trapezoid', dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, supports_out=False, sample_inputs_func=sample_cumulative_trapezoid), OpInfo('unsqueeze', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, check_inplace_batched_forward_grad=False, assert_jit_shape_analysis=True, assert_autodiffed=True, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], sample_inputs_func=sample_unsqueeze), BinaryUfuncInfo('xlogy', aliases=('special.xlogy',), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), promotes_int_to_float=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_one_python_scalar=True, rhs_make_tensor_kwargs=dict(low=0.01)), OpInfo('zero_', op=lambda x: torch.zero_(x.clone()), method_variant=None, inplace_variant=torch.Tensor.zero_, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_gradgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),), sample_inputs_func=sample_inputs_zero_), OpInfo('logsumexp', aliases=('special.logsumexp',), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_fast_mode=False, sample_inputs_func=sample_inputs_logsumexp, reference_inputs_func=reference_inputs_logsumexp), OpInfo('trace', dtypes=all_types_and_complex(), dtypesIfCUDA=all_types_and_complex_and(torch.chalf, torch.bool, torch.half, torch.bfloat16), error_inputs_func=error_inputs_trace, supports_inplace_autograd=False, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_trace), OpInfo('transpose', ref=_numpy_ref_transpose, aliases=('swapdims', 'swapaxes'), assert_jit_shape_analysis=True, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_inplace_batched_forward_grad=False, sample_inputs_func=sample_inputs_transpose_swapdims), OpInfo('T', op=lambda x: x.T, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=sample_inputs_T, error_inputs_func=error_inputs_T), OpInfo('H', op=lambda x: x.H, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=sample_inputs_T), OpInfo('mT', op=lambda x: x.mT, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=sample_inputs_adjoint), OpInfo('mH', op=lambda x: x.mH, aliases=('adjoint',), dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.half, torch.chalf), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), sample_inputs_func=sample_inputs_adjoint), OpInfo('tril', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_forward_ad=True, supports_fwgrad_bwgrad=True, error_inputs_func=error_inputs_tril_triu, sample_inputs_func=sample_inputs_tril_triu), OpInfo('triu', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf), supports_forward_ad=True, supports_fwgrad_bwgrad=True, error_inputs_func=error_inputs_tril_triu, sample_inputs_func=sample_inputs_tril_triu), OpInfo('triu_indices', dtypes=_dispatch_dtypes((torch.int32, torch.int64)), sample_inputs_func=sample_inputs_trilu_indices, ref=lambda h, w, ofs=0, dtype=torch.long, device='cpu': np.array(np.triu_indices(h, ofs, w), dtype=dtype), supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'))), OpInfo('tril_indices', dtypes=_dispatch_dtypes((torch.int32, torch.int64)), sample_inputs_func=sample_inputs_trilu_indices, ref=lambda h, w, ofs=0, dtype=torch.long, device='cpu': np.array(np.tril_indices(h, ofs, w), dtype=dtype), supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'))), OpInfo('kron', dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16), gradcheck_fast_mode=True, supports_inplace_autograd=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_kron, decorators=(DecorateInfo(unittest.expectedFailure, 'TestMeta', 'test_dispatch_symbolic_meta_outplace_all_strides'),)), OpInfo('inner', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_inner), OpInfo('tensordot', dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), dtypesIfROCM=floating_and_complex_types_and(torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, sample_inputs_func=sample_inputs_tensordot, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'),)), OpInfo('to_sparse', op=lambda x, *args: x.to_sparse(*args), sample_inputs_func=sample_inputs_to_sparse, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), backward_dtypes=floating_types(), backward_dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_sparse_csr=True, supports_sparse_csc=True, check_batched_grad=False, check_batched_gradgrad=False, skips=(DecorateInfo(unittest.skip(''), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Allowed exception'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Allowed exception'), 'TestCompositeCompliance', 'test_backward'), DecorateInfo(unittest.skip('Allowed exception'), 'TestTags', 'test_tags'), DecorateInfo(unittest.skip('csr.to_sparse(1) not implemented. Skipped!'), 'TestSparseCSR', 'test_sparse_csr_consistency'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values', dtypes=[torch.bool], active_if=TEST_WITH_ROCM))), OpInfo('logcumsumexp', dtypes=floating_and_complex_types_and(torch.bfloat16, torch.half), backward_dtypes=floating_and_complex_types_and(torch.bfloat16), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_forward_mode_AD', dtypes=[torch.complex128]), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_fn_fwgrad_bwgrad', dtypes=[torch.complex128])), sample_inputs_func=sample_inputs_logcumsumexp, error_inputs_func=error_inputs_logcumsumexp), UnaryUfuncInfo('sigmoid', aliases=('special.expit', 'nn.functional.sigmoid'), aten_backward_name='sigmoid_backward', ref=reference_sigmoid if TEST_SCIPY else None, decorators=(precisionOverride({torch.float16: 0.01, torch.complex64: 0.1, torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.complex64, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.chalf, torch.complex64, torch.cdouble])), dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.complex32, torch.bool, torch.half, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, assert_autodiffed=True, reference_numerics_filter=NumericsFilter(condition=lambda x: close_to_int(x / (math.pi * 1j)) if x.is_complex() else x.new_tensor(False, dtype=torch.bool), safe_val=0)), UnaryUfuncInfo('digamma', ref=scipy.special.digamma if TEST_SCIPY else None, aliases=('special.psi', 'special.digamma'), decorators=(precisionOverride({torch.float16: 0.5}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True), UnaryUfuncInfo('erf', ref=scipy.special.erf if TEST_SCIPY else None, aliases=('special.erf',), decorators=(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped! sparse backward not supported'), 'TestSparseUnaryUfuncs', 'test_sparse_fn_grad'),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, assert_jit_shape_analysis=True, supports_sparse=True, supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True), UnaryUfuncInfo('erfc', ref=scipy.special.erfc if TEST_SCIPY else None, aliases=('special.erfc',), decorators=(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.01}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), assert_autodiffed=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True), UnaryUfuncInfo('erfinv', ref=scipy.special.erfinv if TEST_SCIPY else None, aliases=('special.erfinv',), decorators=(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.01, torch.float32: 0.0001}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half, torch.bfloat16), supports_sparse_csr=True, supports_sparse_csc=True, supports_sparse_bsr=True, supports_sparse_bsc=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, domain=(-1, 1), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', active_if=TEST_SCIPY and version.parse(scipy.__version__) < version.parse('1.4.0')), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', active_if=TEST_SCIPY and version.parse(scipy.__version__) < version.parse('1.4.0')), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', active_if=TEST_SCIPY and version.parse(scipy.__version__) < version.parse('1.4.0')))), OpInfo('nn.functional.smooth_l1_loss', ref=reference_smooth_l1_loss, sample_inputs_func=sample_inputs_smooth_l1_loss, dtypes=floating_types_and(torch.float16, torch.bfloat16), backward_dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16), backward_dtypesIfCUDA=floating_types_and(torch.float16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'),)), OpInfo('nn.functional.l1_loss', ref=loss_reference_reduction_wrapper(lambda input, target: np.abs(input - target)), sample_inputs_func=sample_inputs_l1_loss, error_inputs_func=error_inputs_l1_loss, dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),)), UnaryUfuncInfo('lgamma', ref=reference_lgamma if TEST_SCIPY else None, aliases=('special.gammaln',), decorators=(precisionOverride({torch.float16: 0.7}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.bool, torch.half), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS)), reference_numerics_filter=NumericsFilter(condition=lambda x: x < 0.1, safe_val=1)), OpInfo('logdet', dtypes=floating_and_complex_types(), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_linalg_det_logdet_slogdet, decorators=[skipCUDAIfNoMagma, skipCPUIfNoLapack]), OpInfo('log_softmax', aliases=('special.log_softmax', 'nn.functional.log_softmax'), supports_out=True, aten_backward_name='_log_softmax_backward_data', dtypes=floating_types_and(torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_softmax_variant, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True), OpInfo('log_softmax', variant_test_name='with_dtype', aliases=('special.log_softmax', 'nn.functional.log_softmax'), supports_out=True, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=partial(sample_inputs_softmax_variant, with_dtype=True), supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True), UnaryUfuncInfo('logit', aten_backward_name='logit_backward', ref=scipy.special.logit if TEST_SCIPY else None, domain=(0, 1), aliases=('special.logit',), supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_float=True, decorators=(precisionOverride({torch.bfloat16: 0.5, torch.float16: 0.5}),), dtypes=all_types_and(torch.bool, torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_logit), OpInfo('where', op=lambda self, condition, other: torch.where(condition, self, other), ref=lambda self, condition, other: np.where(condition, self, other), sample_inputs_func=sample_inputs_where, reference_inputs_func=reference_inputs_where, error_inputs_func=error_inputs_where, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, decorators=(DecorateInfo(onlyCUDA, 'TestCommon', 'test_errors'),), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit')), dtypes=all_types_and_complex_and(torch.bool, torch.half, torch.bfloat16, torch.chalf)), OpInfo('nonzero', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), sample_inputs_func=sample_inputs_nonzero, supports_autograd=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestOperatorSignatures', 'test_get_torch_func_signature_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values', dtypes=[torch.bool], active_if=TEST_WITH_ROCM))), OpInfo('nonzero_static', dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16, torch.chalf), sample_inputs_func=sample_inputs_nonzero_static, supports_out=False, supports_autograd=False, decorators=[onlyCPU], skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestDTensorOps', 'test_dtensor_op_db'), DecorateInfo(unittest.expectedFailure, 'TestInductorOpInfo', 'test_comprehensive'), DecorateInfo(unittest.expectedFailure, 'TestVmapOperatorsOpInfo', 'test_op_has_batch_rule'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values', dtypes=[torch.bool], active_if=TEST_WITH_ROCM))), UnaryUfuncInfo('jiterator_unary', op=torch.cuda.jiterator._create_jit_fn('template <typename T> T unary(T x) { return x * x + x; }'), ref=lambda x: x * x + x, dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.bool), supports_out=False, supports_autograd=False, decorators=[onlyCUDA, DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestUnaryUfuncs', 'test_reference_numerics_hard'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestUnaryUfuncs', 'test_reference_numerics_normal'), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.01)}), 'TestUnaryUfuncs', 'test_reference_numerics_small')], skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('skip'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.bool]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_hard', dtypes=[torch.bool]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', dtypes=[torch.bool]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.complex64], active_if=TEST_WITH_ROCM), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'))), BinaryUfuncInfo('jiterator_binary', op=torch.cuda.jiterator._create_jit_fn('template <typename T> T binary(T x, T y, T alpha) { return x + alpha * y; }', alpha=1), ref=lambda input, other, *, alpha=1: np.add(input, other) if alpha == 1 else np.add(input, np.multiply(alpha, other)), dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.bool), sample_inputs_func=partial(sample_inputs_jiterator, num_inputs=2, alpha=-3.14), supports_out=False, supports_autograd=False, supports_rhs_python_scalar=False, decorators=[onlyCUDA], skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('skip'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'))), OpInfo('jiterator_4inputs_with_extra_args', op=torch.cuda.jiterator._create_jit_fn('template <typename T> T binary(T i0, T i1, T i2, T i3, T alpha, T beta) { return alpha * i0 + beta * i1 + i2 + i3; }', alpha=1, beta=1), ref=lambda i0, i1, i2, i3, *, alpha=1, beta=1: alpha * i0 + beta * i1 + i2 + i3, dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.bool), sample_inputs_func=partial(sample_inputs_jiterator, num_inputs=4, alpha=3.14, beta=-4.2), supports_out=False, supports_autograd=False, decorators=[onlyCUDA], skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('skip'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'))), BinaryUfuncInfo('jiterator_binary_return_by_ref', op=torch.cuda.jiterator._create_multi_output_jit_fn('\n            template <typename T>\n            void binary_return_by_ref(T i0, T i1, T& out0) {\n                out0 = i0 + i1;\n            }\n            ', num_outputs=1), ref=lambda i0, i1: i0 + i1, dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.bool), sample_inputs_func=partial(sample_inputs_jiterator, num_inputs=2, alpha=-0.42), supports_out=False, supports_autograd=False, supports_rhs_python_scalar=False, decorators=[onlyCUDA], skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('skip'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'))), OpInfo('jiterator_2inputs_2outputs', op=torch.cuda.jiterator._create_multi_output_jit_fn('\n            template <typename T>\n            void binary_2outputs(T i0, T i1, T& out0, T& out1) {\n                out0 = i0 + i1;\n                out1 = i0 - i1;\n            }\n            ', num_outputs=2), ref=lambda i0, i1, *, alpha=1: (i0 + i1, i0 - i1), dtypes=all_types_and_complex_and(torch.bfloat16, torch.float16, torch.bool), sample_inputs_func=partial(sample_inputs_jiterator, num_inputs=2), supports_out=False, supports_autograd=False, decorators=[onlyCUDA], skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('skip'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestCudaFuserOpInfo'))), OpInfo('norm', sample_inputs_func=sample_inputs_norm, dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), gradcheck_fast_mode=True, check_batched_forward_grad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),)), OpInfo('norm', variant_test_name='nuc', sample_inputs_func=sample_inputs_norm_nuc, decorators=[skipCUDAIfNoMagmaAndNoCusolver, skipCPUIfNoLapack], check_batched_gradgrad=False, check_batched_forward_grad=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_and_complex_types(), dtypesIfCUDA=floating_and_complex_types(), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.complex64, torch.float32)), DecorateInfo(unittest.skip('Skipped!'), 'TestFakeTensor', 'test_fake_crossref_backward_amp', device_type='cuda', dtypes=[torch.float32], active_if=TEST_WITH_ROCM), DecorateInfo(unittest.skip('Skipped!'), 'TestFakeTensor', 'test_fake_crossref_backward_no_amp', device_type='cuda', dtypes=[torch.float32], active_if=TEST_WITH_ROCM))), OpInfo('norm', variant_test_name='fro', sample_inputs_func=sample_inputs_norm_fro, dtypes=floating_and_complex_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.0001, rtol=0.01)}), 'TestConsistency', 'test_output_match'), DecorateInfo(unittest.skip('Skipped!'), 'TestSchemaCheckModeOpInfo', 'test_schema_correctness', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.complex64, torch.float32)))), OpInfo('norm', variant_test_name='inf', sample_inputs_func=sample_inputs_norm_inf, dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, gradcheck_fast_mode=False, skips=(DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.002, rtol=0.001)}), 'TestInductorOpInfo', 'test_comprehensive', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)))), OpInfo('t', sample_inputs_func=sample_inputs_t, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, check_inplace_batched_forward_grad=False, autodiff_fusible_nodes=[], autodiff_nonfusible_nodes=[], dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), assert_autodiffed=True, error_inputs_func=error_inputs_t), OpInfo('nn.functional.dropout', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.dropout, input, *args, **kwargs), dtypes=floating_types_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view', device_type='cuda'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu')), supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, supports_out=False, sample_inputs_func=sample_inputs_dropout, inplace_variant=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.dropout, input, *args, **kwargs, inplace=True)), OpInfo('native_dropout_backward', op=torch.ops.aten.native_dropout_backward.default, aten_name='native_dropout_backward', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_dropout_backward, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestLazyOpInfo', 'test_dispatched_to_lazy'), DecorateInfo(unittest.expectedFailure, 'TestLazyOpInfo', 'test_correctness'), DecorateInfo(unittest.expectedFailure, 'TestLazyOpInfo', 'test_correctness_with_reusing_ir'))), OpInfo('nn.functional.dropout2d', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.dropout2d, input, *args, **kwargs), dtypes=floating_types_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu')), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, check_batched_forward_grad=False, sample_inputs_func=partial(sample_inputs_dropout, valid_input_dim=(3, 4)), inplace_variant=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.dropout2d, input, *args, **kwargs, inplace=True)), OpInfo('nn.functional.dropout3d', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.dropout3d, input, *args, **kwargs), dtypes=floating_types_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu')), supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, check_batched_forward_grad=False, sample_inputs_func=partial(sample_inputs_dropout, valid_input_dim=(4, 5)), inplace_variant=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.dropout3d, input, *args, **kwargs, inplace=True)), OpInfo('nn.functional.alpha_dropout', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.alpha_dropout, input, *args, **kwargs), dtypes=floating_types_and(torch.float16, torch.bfloat16), gradcheck_wrapper=wrapper_set_seed, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, sample_inputs_func=sample_inputs_dropout, check_batched_forward_grad=False, inplace_variant=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.alpha_dropout, input, *args, **kwargs, inplace=True), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_compare_cpu', device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'))), OpInfo('nn.functional.feature_alpha_dropout', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs), variant_test_name='with_train', dtypes=floating_types_and(torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_inplace_forward_mode_AD'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu')), gradcheck_fast_mode=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, sample_inputs_func=partial(sample_inputs_dropout, train=True, valid_input_dim=(4, 5)), inplace_variant=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs, inplace=True)), OpInfo('nn.functional.feature_alpha_dropout', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs), variant_test_name='without_train', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit')), gradcheck_wrapper=wrapper_set_seed, supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, sample_inputs_func=partial(sample_inputs_dropout, train=False), inplace_variant=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.feature_alpha_dropout, input, *args, **kwargs, inplace=True)), OpInfo('nn.functional.one_hot', ref=reference_one_hot, supports_out=False, dtypes=_dispatch_dtypes((torch.int64,)), sample_inputs_func=sample_inputs_one_hot), OpInfo('nn.functional.embedding', aten_backward_name='embedding_dense_backward', op=lambda weight, idx, **kwargs: torch.nn.functional.embedding(idx, weight, **kwargs), dtypes=floating_types_and(torch.bfloat16, torch.float16), sample_inputs_func=sample_inputs_embedding, error_inputs_func=error_inputs_embedding, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view', device_type='cuda'), DecorateInfo(unittest.skip('Allowed exemption'), 'TestCompositeCompliance', 'test_operator')), supports_expanded_weight=True, supports_out=False), OpInfo('nn.functional.embedding_bag', op=lambda weight, idx, **kwargs: torch.nn.functional.embedding_bag(idx, weight, **kwargs), dtypes=floating_types_and(torch.bfloat16, torch.float16), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16), backward_dtypesIfCUDA=floating_types_and(torch.float16), sample_inputs_func=sample_inputs_embedding_bag, skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Allowed exemption'), 'TestCompositeCompliance', 'test_operator')), gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, supports_out=False, supports_gradgrad=False), OpInfo('nn.functional.multi_head_attention_forward', op=lambda input, *args, **kwargs: wrapper_set_seed(torch.nn.functional.multi_head_attention_forward, input, *args, **kwargs), dtypes=floating_types_and(torch.bfloat16, torch.float16), sample_inputs_func=sample_inputs_multi_head_attention_forward, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples', dtypes=(torch.float32,)), DecorateInfo(toleranceOverride({torch.float32: tol(atol=0.005, rtol=0)}), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('Skipped!'), 'TestInductorOpInfo', 'test_comprehensive'), DecorateInfo(unittest.skip('Skipped!'), 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestNormalizeOperators', 'test_normalize_operator_exhaustive'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_forward_ad'), DecorateInfo(unittest.skip('Skipped!'), 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Skipped - baddbmm decomp does not have enough precision for 16-bit float'), 'TestDecomp', 'test_comprehensive', dtypes=(torch.bfloat16, torch.float16)), DecorateInfo(unittest.skip('Skipped - baddbmm decomp does not have enough precision for 16-bit float'), 'TestDecomp', 'test_quick', dtypes=(torch.bfloat16, torch.float16))), supports_out=False, supports_gradgrad=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, gradcheck_fast_mode=True), UnaryUfuncInfo('nn.functional.softplus', aten_backward_name='softplus_backward', ref=reference_softplus, sample_kwargs=lambda device, dtype, input: ({'beta': 3, 'threshold': 0.2}, {'beta': 3, 'threshold': 0.2}), sample_inputs_func=partial(sample_inputs_elementwise_unary, op_kwargs={'beta': 3, 'threshold': 0.2}), supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.bfloat16, torch.float16), decorators=(DecorateInfo(toleranceOverride({torch.half: tol(atol=0.01, rtol=0.01), torch.bfloat16: tol(atol=0.01, rtol=0.01)}), 'TestUnaryUfuncs'),)), OpInfo('nn.functional.mse_loss', aten_backward_name='mse_loss_backward', ref=loss_reference_reduction_wrapper(lambda input, target: (input - target) ** 2), sample_inputs_func=sample_inputs_loss, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.float16), backward_dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16), backward_dtypesIfCUDA=floating_types_and(torch.bfloat16, torch.float16), skips=(DecorateInfo(unittest.expectedFailure, 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),)), OpInfo('nn.functional.grid_sample', dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_grid_sample, reference_inputs_func=reference_inputs_grid_sample, supports_gradgrad=False, gradcheck_nondet_tol=1e-15), OpInfo('grid_sampler_2d', dtypes=floating_types(), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_grid_sampler_2d, supports_gradgrad=False, gradcheck_nondet_tol=1e-15), OpInfo('argwhere', ref=np.argwhere, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_autograd=False, sample_inputs_func=sample_inputs_argwhere, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_non_standard_bool_values', dtypes=[torch.bool], active_if=TEST_WITH_ROCM),)), ReductionOpInfo('all', identity=True, supports_autograd=False, result_dtype=torch.bool, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), ref=reference_reduction_numpy(np.all), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_result_dtype', dtypes=[torch.uint8]),)), ReductionOpInfo('any', identity=False, supports_autograd=False, result_dtype=torch.bool, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), ref=reference_reduction_numpy(np.any), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_result_dtype', dtypes=[torch.uint8]),)), ReductionOpInfo('amax', nan_policy='propagate', supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), ref=reference_reduction_numpy(np.amax), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim')), error_inputs_func=error_inputs_aminmax_amax_amin), ReductionOpInfo('amin', nan_policy='propagate', supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), ref=reference_reduction_numpy(np.amin), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim')), error_inputs_func=error_inputs_aminmax_amax_amin), ReductionOpInfo('argmax', supports_multiple_dims=False, supports_autograd=False, assert_jit_shape_analysis=True, result_dtype=torch.int64, dtypes=all_types_and(torch.float16, torch.bfloat16), ref=reference_reduction_numpy(np.argmax, supports_keepdims=False)), ReductionOpInfo('argmin', supports_multiple_dims=False, supports_autograd=False, result_dtype=torch.int64, dtypes=all_types_and(torch.float16, torch.bfloat16), ref=reference_reduction_numpy(np.argmin, supports_keepdims=False)), ReductionOpInfo('count_nonzero', identity=0, supports_out=False, supports_autograd=False, result_dtype=torch.int64, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_reduction_count_nonzero, ref=reference_reduction_numpy(np.count_nonzero), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_default_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_none_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_single_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_multi_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_multi_unsorted_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_offbounds_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'))), ReductionOpInfo('mean', nan_policy='propagate', supports_forward_ad=True, supports_fwgrad_bwgrad=True, supports_out=False, assert_autodiffed=True, assert_jit_shape_analysis=True, promotes_int_to_float=True, dtypes=floating_and_complex_types_and(torch.float16, torch.bfloat16), ref=reference_reduction_numpy(np.mean), error_inputs_func=error_inputs_mean, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_default_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=[torch.float16]), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_extremal_values', device_type='cuda', dtypes=[torch.complex64]))), ReductionOpInfo('nanmean', nan_policy='omit', assert_autodiffed=True, promotes_int_to_float=True, supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, dtypes=floating_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_nan_reduction(supports_multiple_dims=True), ref=reference_reduction_numpy(np.nanmean), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=[torch.float16]), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_duplicate_values', device_type='cuda', dtypes=[torch.float16]), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_extremal_values', device_type='cuda', dtypes=[torch.complex64]))), ReductionOpInfo('std', nan_policy='propagate', supports_out=True, complex_to_real=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, promotes_int_to_float=True, check_batched_forward_grad=False, dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var, ref=reference_std_var(np.std), generate_args_kwargs=generate_std_var_kwargs, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_default_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=(torch.float16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_duplicate_values', dtypes=(torch.float16,)))), ReductionOpInfo('std', variant_test_name='unbiased', nan_policy='propagate', supports_out=False, complex_to_real=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, promotes_int_to_float=True, check_batched_forward_grad=False, dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var_unbiased, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'))), ReductionOpInfo('var', nan_policy='propagate', supports_out=True, assert_autodiffed=True, promotes_int_to_float=True, complex_to_real=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var, ref=reference_std_var(np.var), generate_args_kwargs=generate_std_var_kwargs, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_default_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_duplicate_values'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_large_input'))), ReductionOpInfo('var', variant_test_name='unbiased', nan_policy='propagate', supports_out=False, complex_to_real=True, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_autodiffed=True, promotes_int_to_float=True, check_batched_forward_grad=False, dtypes=floating_and_complex_types_and(torch.half, torch.bfloat16), dtypesIfCUDA=floating_and_complex_types_and(torch.half, torch.bfloat16), sample_inputs_func=sample_inputs_std_var_unbiased, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'))), ReductionOpInfo('prod', identity=1, nan_policy='propagate', supports_multiple_dims=False, gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_int64=True, gradcheck_nondet_tol=GRADCHECK_NONDET_TOL, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_prod, ref=prod_numpy, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_default_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_none'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_none_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=[torch.float16, torch.complex64]), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_duplicate_values', dtypes=[torch.uint8, torch.float16, torch.complex64]), DecorateInfo(unittest.skip('Skipped!'), 'TestOperators', 'test_reduction_all', dtypes=[torch.float16]))), ReductionOpInfo('sum', identity=0, nan_policy='propagate', supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, promotes_int_to_int64=True, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), ref=reference_reduction_numpy(np.sum), error_inputs_sparse_func=error_inputs_sparse_reduction_sum, sample_inputs_sparse_coo_func=partial(sample_inputs_sparse_reduction_sum, layout=torch.sparse_coo), sample_inputs_sparse_csr_func=partial(sample_inputs_sparse_reduction_sum, layout=torch.sparse_csr), sample_inputs_sparse_csc_func=partial(sample_inputs_sparse_reduction_sum, layout=torch.sparse_csc), sample_inputs_sparse_bsr_func=partial(sample_inputs_sparse_reduction_sum, layout=torch.sparse_bsr), sample_inputs_sparse_bsc_func=partial(sample_inputs_sparse_reduction_sum, layout=torch.sparse_bsc), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_default_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=[torch.float16]), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_duplicate_values', dtypes=[torch.float16]), DecorateInfo(unittest.skip('Skipped!'), 'TestOperators', 'test_reduction_all', dtypes=[torch.float32]))), ReductionOpInfo('nansum', identity=0, nan_policy='omit', supports_out=True, promotes_int_to_int64=True, supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_nan_reduction(supports_multiple_dims=True), ref=reference_reduction_numpy(np.nansum), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=[torch.float16]))), OpInfo('nn.functional.ctc_loss', dtypes=floating_types(), supports_out=False, sample_inputs_func=sample_inputs_ctc_loss, skips=(DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_grad', dtypes=(torch.float64,)), DecorateInfo(unittest.expectedFailure, 'TestBwdGradients', 'test_fn_gradgrad', dtypes=(torch.float64,)), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)), DecorateInfo(unittest.skip('Fails with ASAN'), 'TestProxyTensorOpInfo', 'test_make_fx_fake_exhaustive', active_if=TEST_WITH_ASAN))), OpInfo('nn.functional.cosine_embedding_loss', dtypes=all_types_and(torch.half, torch.bfloat16, torch.bool), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_cosine_embedding_loss), OpInfo('nn.functional.nll_loss', dtypes=floating_types_and(torch.bfloat16), dtypesIfCUDA=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, sample_inputs_func=sample_inputs_nll_loss, supports_forward_ad=True, supports_fwgrad_bwgrad=True, assert_jit_shape_analysis=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),)), OpInfo('nn.functional.gaussian_nll_loss', dtypes=floating_types_and(torch.half, torch.bfloat16), gradcheck_fast_mode=True, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_gaussian_nll_loss, error_inputs_func=error_inputs_gaussian_nll_loss, skips=(DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_backward'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_forward_ad'), DecorateInfo(unittest.expectedFailure, 'TestCompositeCompliance', 'test_operator'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)))), OpInfo('nn.functional.hinge_embedding_loss', dtypes=floating_types_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_hinge_embedding_loss, error_inputs_func=error_inputs_hinge_embedding_loss, reference_inputs_func=reference_inputs_hinge_embedding_loss), OpInfo('nn.functional.huber_loss', aten_backward_name='huber_loss_backward', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, sample_inputs_func=sample_inputs_huber_loss, error_inputs_func=error_inputs_huber_loss, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),)), OpInfo('nn.functional.pdist', ref=reference_pdist, sample_inputs_func=sample_inputs_pdist, dtypes=floating_types(), supports_out=False, supports_gradgrad=False, skips=(DecorateInfo(unittest.skip('Unsupported on MPS for now'), 'TestCommon', 'test_numpy_ref_mps'),)), OpInfo('nn.functional.poisson_nll_loss', dtypes=all_types_and(torch.half, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_poisson_nll_loss, error_inputs_func=error_inputs_poisson_nll_loss), OpInfo('argsort', dtypes=all_types_and(torch.bool, torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_sort, supports_out=False, supports_autograd=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32,)),)), OpInfo('repeat_interleave', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16, torch.chalf), backward_dtypesIfCUDA=floating_and_complex_types_and(torch.float16, torch.bfloat16, torch.chalf), sample_inputs_func=sample_inputs_repeat_interleave, supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)),)), OpInfo('nn.functional.pairwise_distance', ref=lambda a, b, p=2.0, eps=1e-06, keepdim=False: np.sum(np.abs(a - b + eps) ** p, axis=-1, keepdims=keepdim) ** (1 / p), sample_inputs_func=sample_inputs_pairwise_distance, dtypes=all_types_and_complex_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)),)), OpInfo('nn.functional.pixel_shuffle', sample_inputs_func=sample_inputs_pixel_shuffle, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)),)), OpInfo('nn.functional.pixel_unshuffle', sample_inputs_func=sample_inputs_pixel_unshuffle, dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', dtypes=(torch.float32, torch.complex64)),)), OpInfo('nn.functional.kl_div', sample_inputs_func=sample_inputs_kl_div, dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True), OpInfo('diagflat', ref=lambda input, offset=0: np.diagflat(input, k=offset), sample_inputs_func=sample_inputs_diagflat, dtypes=all_types_and_complex_and(torch.bool, torch.bfloat16, torch.float16), dtypesIfCUDA=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), supports_out=False, supports_forward_ad=True, supports_fwgrad_bwgrad=True, check_batched_forward_grad=False), OpInfo('scatter_reduce', variant_test_name='sum', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_scatter_reduce), OpInfo('scatter_reduce', variant_test_name='prod', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), sample_inputs_func=sample_inputs_scatter_reduce, skips=(DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_inplace_forward_mode_AD'), DecorateInfo(unittest.expectedFailure, 'TestFwdGradients', 'test_fn_fwgrad_bwgrad'))), OpInfo('scatter_reduce', variant_test_name='mean', dtypes=all_types_and(torch.float16, torch.bfloat16), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_scatter_reduce), OpInfo('scatter_reduce', variant_test_name='amin', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_scatter_reduce), OpInfo('scatter_reduce', variant_test_name='amax', dtypes=all_types_and(torch.float16, torch.bfloat16, torch.bool), dtypesIfCUDA=all_types_and(torch.float16, torch.bfloat16), supports_forward_ad=True, check_batched_forward_grad=False, supports_fwgrad_bwgrad=True, sample_inputs_func=sample_inputs_scatter_reduce), OpInfo('_segment_reduce', aten_name='segment_reduce', variant_test_name='lengths', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_gradgrad=False, sample_inputs_func=sample_inputs_segment_reduce, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='cuda'),)), OpInfo('_segment_reduce', aten_name='segment_reduce', variant_test_name='offsets', dtypes=floating_types_and(torch.float16, torch.bfloat16), supports_out=False, supports_gradgrad=False, sample_inputs_func=partial(sample_inputs_segment_reduce, mode='offsets'), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit', device_type='cuda'),))]
op_db += opinfo.definitions.op_db
python_ref_db = [ElementwiseUnaryPythonRefInfo('_refs.abs', torch_opinfo_name='abs', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=[torch.int8], active_if=TEST_WITH_ASAN),)), ElementwiseUnaryPythonRefInfo('_refs.acos', torch_opinfo_name='acos', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]))), ElementwiseUnaryPythonRefInfo('_refs.acosh', torch_opinfo_name='acosh', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.asin', torch_opinfo_name='asin', decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=1e-05, rtol=0.001)}), 'TestUnaryUfuncs', device_type='cuda'), precisionOverride({torch.bfloat16: 0.01})], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.asinh', torch_opinfo_name='asinh', decorators=(precisionOverride({torch.bfloat16: 0.05}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cdouble], active_if=IS_WINDOWS))), PythonRefInfo('_refs.lerp', torch_opinfo_name='lerp'), PythonRefInfo('_refs.ones', torch_opinfo_name='ones', skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.zeros', torch_opinfo_name='zeros', skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.cauchy', torch_opinfo_name='cauchy', decorators=(DecorateInfo(unittest.skip('TODO: RuntimeError: no _refs support for torch.rand_like'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: cauchy is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: cauchy is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('Expected: cauchy is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'))), PythonRefInfo('_refs.exponential', torch_opinfo_name='exponential', supports_out=True, decorators=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_meta', dtypes=(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.skip('TODO: RuntimeError: no _refs support for torch.rand_like'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: exponential is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: exponential is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('Expected: exponential is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'))), PythonRefInfo('_refs.geometric', torch_opinfo_name='geometric', supports_out=True, decorators=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_dtypes'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_meta', dtypes=(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64)), DecorateInfo(unittest.skip('TODO: RuntimeError: no _refs support for torch.rand_like'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: geometric is not comparable'), 'TestCommon', 'test_python_ref_executor', device_type='cuda'), DecorateInfo(unittest.skip('Expected: geometric is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: geometric is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: geometric is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'))), PythonRefInfo('_refs.log_normal', torch_opinfo_name='log_normal', supports_out=True, decorators=(DecorateInfo(unittest.skip('TODO: RuntimeError: no _refs support for torch.rand_like'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: log_normal is not comparable'), 'TestCommon', 'test_python_ref_executor', device_type='cuda'), DecorateInfo(unittest.skip('Expected: log_normal is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: log_normal is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: log_normal is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'))), PythonRefInfo('_refs.normal', torch_opinfo_name='normal', supports_out=True, decorators=(DecorateInfo(unittest.skip('TODO: RuntimeError: no _refs support for torch.rand_like'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip("make_traced() doesn't set seed properly!"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.normal', torch_opinfo_name='normal', torch_opinfo_variant_name='number_mean', supports_out=True, decorators=(DecorateInfo(unittest.skip('TODO: RuntimeError: no _refs support for torch.rand_like'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip("make_traced() doesn't set seed properly!"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.normal_', op=torch.Tensor.normal_, torch_opinfo_name='normal', torch_opinfo_variant_name='in_place', supports_out=False, decorators=(DecorateInfo(unittest.skip('TODO: RuntimeError: no _refs support for torch.rand_like'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: normal is not comparable'), 'TestDecomp', 'test_comprehensive'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'), DecorateInfo(unittest.skip("make_traced() doesn't set seed properly!"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.arange', torch_opinfo_name='arange', skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.linspace', torch_opinfo_name='linspace', skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64), device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64), device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', dtypes=(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), device_type='cuda'))), PythonRefInfo('_refs.linspace', torch_opinfo_name='linspace', torch_opinfo_variant_name='tensor_overload', skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64), device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.int8, torch.uint8, torch.int16, torch.int32, torch.int64), device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', dtypes=(torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64), device_type='cuda'))), PythonRefInfo('_refs.logspace', torch_opinfo_name='logspace', skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'))), PythonRefInfo('_refs.logspace', torch_opinfo_name='logspace', torch_opinfo_variant_name='tensor_overload', skips=(DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', dtypes=(torch.int16, torch.int32, torch.int64), device_type='cuda'))), PythonRefInfo('_refs.meshgrid', torch_opinfo_name='meshgrid', torch_opinfo_variant_name='variadic_tensors'), PythonRefInfo('_refs.take_along_dim', torch_opinfo_name='take_along_dim', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'),)), PythonRefInfo('_refs.to', torch_opinfo_name='to'), PythonRefInfo('_refs.triu', torch_opinfo_name='triu'), PythonRefInfo('_refs.tril', torch_opinfo_name='tril'), PythonRefInfo('_refs.triu_indices', torch_opinfo_name='triu_indices', validate_view_consistency=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'))), PythonRefInfo('_refs.tril_indices', torch_opinfo_name='tril_indices', validate_view_consistency=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_variant_consistency_eager'), DecorateInfo(unittest.skip('Skipped!'), 'TestJit', 'test_variant_consistency_jit'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'))), PythonRefInfo('_refs.meshgrid', torch_opinfo_name='meshgrid', torch_opinfo_variant_name='list_of_tensors'), PythonRefInfo('_refs.movedim', aliases=('moveaxis',), torch_opinfo_name='movedim'), PythonRefInfo('_refs.bucketize', torch_opinfo_name='bucketize', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor'),)), PythonRefInfo('_refs.equal', torch_opinfo_name='equal', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_meta'),)), ElementwiseUnaryPythonRefInfo('_refs.atan', torch_opinfo_name='atan', decorators=(precisionOverride({torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', active_if=TEST_WITH_ROCM, device_type='cuda', dtypes=[torch.complex64, torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', active_if=TEST_WITH_ROCM, device_type='cuda', dtypes=[torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.atanh', torch_opinfo_name='atanh', decorators=(precisionOverride({torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=[torch.cfloat], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', active_if=TEST_WITH_ROCM, device_type='cuda', dtypes=[torch.complex128]))), ElementwiseUnaryPythonRefInfo('_refs.bitwise_not', torch_opinfo_name='bitwise_not'), ElementwiseUnaryPythonRefInfo('_refs.ceil', torch_opinfo_name='ceil'), PythonRefInfo('_refs.item', torch_opinfo_name='item', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_meta'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'))), ElementwiseUnaryPythonRefInfo('_refs.conj_physical', torch_opinfo_name='conj_physical'), ElementwiseUnaryPythonRefInfo('_refs.cos', torch_opinfo_name='cos', decorators=(precisionOverride({torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cfloat, torch.cdouble), device_type='cpu', active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cdouble,), device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=(torch.chalf,), active_if=IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.cosh', torch_opinfo_name='cosh', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.int8]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cuda', dtypes=(torch.chalf,), active_if=IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.digamma', torch_opinfo_name='digamma'), ElementwiseUnaryPythonRefInfo('_refs.erf', torch_opinfo_name='erf'), ElementwiseUnaryPythonRefInfo('_refs.erfinv', torch_opinfo_name='erfinv', decorators=(precisionOverride({torch.float16: 0.01, torch.bfloat16: 0.01, torch.float32: 0.0001}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', active_if=TEST_SCIPY and version.parse(scipy.__version__) < version.parse('1.4.0')), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', active_if=TEST_SCIPY and version.parse(scipy.__version__) < version.parse('1.4.0')), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', active_if=TEST_SCIPY and version.parse(scipy.__version__) < version.parse('1.4.0')))), ElementwiseUnaryPythonRefInfo('_refs.erfc', torch_opinfo_name='erfc'), ElementwiseUnaryPythonRefInfo('_refs.exp', torch_opinfo_name='exp', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]))), ElementwiseUnaryPythonRefInfo('_refs.expm1', torch_opinfo_name='expm1'), ElementwiseUnaryPythonRefInfo('_refs.exp2', torch_opinfo_name='exp2', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]))), ElementwiseUnaryPythonRefInfo('_refs.fill', torch_opinfo_name='fill', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.floor', torch_opinfo_name='floor'), ElementwiseUnaryPythonRefInfo('_refs.frac', torch_opinfo_name='frac', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.bfloat16, torch.float16, torch.float32, torch.float64)),)), ElementwiseUnaryPythonRefInfo('_refs.imag', torch_opinfo_name='imag'), ElementwiseUnaryPythonRefInfo('_refs.isfinite', torch_opinfo_name='isfinite', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.isinf', torch_opinfo_name='isinf', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.isposinf', torch_opinfo_name='isposinf', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.isneginf', torch_opinfo_name='isneginf', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.isnan', torch_opinfo_name='isnan', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.isreal', torch_opinfo_name='isreal', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.i0', torch_opinfo_name='i0', decorators=(precisionOverride({torch.bfloat16: 0.3, torch.float16: 0.5}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.int8,)),)), ElementwiseUnaryPythonRefInfo('_refs.lgamma', torch_opinfo_name='lgamma', decorators=(precisionOverride({torch.float16: 0.7}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.float32, torch.float64], active_if=IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.special.multigammaln', torch_opinfo_name='mvlgamma', torch_opinfo_variant_name='mvlgamma_p_1', skips=skips_mvlgamma()), ElementwiseUnaryPythonRefInfo('_refs.special.multigammaln', torch_opinfo_name='mvlgamma', torch_opinfo_variant_name='mvlgamma_p_3', skips=skips_mvlgamma()), ElementwiseUnaryPythonRefInfo('_refs.special.multigammaln', torch_opinfo_name='mvlgamma', torch_opinfo_variant_name='mvlgamma_p_5', skips=skips_mvlgamma()), ElementwiseUnaryPythonRefInfo('_refs.log', torch_opinfo_name='log', decorators=(precisionOverride({torch.bfloat16: 0.05}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),)), ElementwiseUnaryPythonRefInfo('_refs.log1p', torch_opinfo_name='log1p'), ElementwiseUnaryPythonRefInfo('_refs.log10', torch_opinfo_name='log10', decorators=(precisionOverride({torch.bfloat16: 0.05}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_WINDOWS),)), ElementwiseUnaryPythonRefInfo('_refs.log2', torch_opinfo_name='log2', decorators=(precisionOverride({torch.bfloat16: 0.1}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble]),)), PythonRefInfo('_refs.logsumexp', torch_opinfo_name='logsumexp'), PythonRefInfo('_refs.log_softmax', torch_opinfo_name='log_softmax', torch_opinfo_variant_name='with_dtype'), ElementwiseUnaryPythonRefInfo('_refs.nan_to_num', torch_opinfo_name='nan_to_num'), ElementwiseUnaryPythonRefInfo('_refs.neg', torch_opinfo_name='neg'), ElementwiseUnaryPythonRefInfo('_refs.positive', torch_opinfo_name='positive'), ElementwiseUnaryPythonRefInfo('_refs.real', torch_opinfo_name='real'), ElementwiseUnaryPythonRefInfo('_refs.reciprocal', torch_opinfo_name='reciprocal', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.cfloat, torch.cdouble]),)), ElementwiseUnaryPythonRefInfo('_refs.round', torch_opinfo_name='round', skips=(DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.001, rtol=0.016)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda'), DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.001, rtol=0.016)}), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cuda'))), ElementwiseUnaryPythonRefInfo('_refs.rsqrt', torch_opinfo_name='rsqrt', decorators=(precisionOverride({torch.half: 0.05}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.cfloat, torch.cdouble)), DecorateInfo(unittest.expectedFailure, 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.chalf,)))), ElementwiseUnaryPythonRefInfo('_refs.sigmoid', torch_opinfo_name='sigmoid', aliases=('_refs.special.expit',), handles_complex_extremal_values=False, handles_large_floats=False, decorators=(precisionOverride({torch.float16: 0.01, torch.complex64: 0.1, torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.complex64, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.chalf, torch.complex64, torch.cdouble]))), ElementwiseUnaryPythonRefInfo('_refs.sign', torch_opinfo_name='sign', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),)), ElementwiseUnaryPythonRefInfo('_refs.sgn', torch_opinfo_name='sgn', handles_complex_extremal_values=False, handles_large_floats=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=[torch.bfloat16, torch.float16, torch.float32, torch.float64]),)), ElementwiseUnaryPythonRefInfo('_refs.signbit', torch_opinfo_name='signbit'), ElementwiseUnaryPythonRefInfo('_refs.sin', torch_opinfo_name='sin', decorators=(precisionOverride({torch.bfloat16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cdouble,), device_type='cuda'), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.cfloat, torch.cdouble), device_type='cpu', active_if=IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cfloat, torch.cdouble), device_type='cpu', active_if=IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.sinc', torch_opinfo_name='sinc', decorators=(precisionOverride({torch.bfloat16: 0.01, torch.float16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_small', dtypes=[torch.cfloat]),)), ElementwiseUnaryPythonRefInfo('_refs.sinh', torch_opinfo_name='sinh', decorators=(precisionOverride({torch.float16: 0.01}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.cdouble,)), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.int8]))), PythonRefInfo('_refs.softmax', torch_opinfo_name='softmax', torch_opinfo_variant_name='with_dtype'), ElementwiseUnaryPythonRefInfo('_refs.sqrt', torch_opinfo_name='sqrt', decorators=(precisionOverride({torch.bfloat16: 0.07}), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.01, rtol=0)}), 'TestUnaryUfuncs', 'test_reference_numerics_large')), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=(torch.cfloat, torch.cdouble), active_if=IS_MACOS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.bfloat16,)))), ElementwiseUnaryPythonRefInfo('_refs.square', torch_opinfo_name='square', decorators=(precisionOverride({torch.complex64: 0.0003, torch.bfloat16: 0.3}),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref_executor', dtypes=(torch.complex64,)), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda', dtypes=[torch.cfloat, torch.cdouble]))), ElementwiseUnaryPythonRefInfo('_refs.tan', torch_opinfo_name='tan', decorators=[DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=1e-05)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.tanh', torch_opinfo_name='tanh', decorators=[DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=2e-05)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda')], skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS), DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble], active_if=IS_MACOS or IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.trunc', torch_opinfo_name='trunc'), PythonRefInfo('_refs.special.log_softmax', torch_opinfo_name='log_softmax', torch_opinfo_variant_name='with_dtype', supports_out=False), PythonRefInfo('_refs.special.softmax', torch_opinfo_name='softmax', torch_opinfo_variant_name='with_dtype', supports_out=False), ElementwiseUnaryPythonRefInfo('_refs.special.logit', torch_opinfo_name='logit'), PythonRefInfo('_refs.nn.functional.alpha_dropout', torch_opinfo_name='nn.functional.alpha_dropout', decorators=(DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_python_ref_executor', device_type='cuda'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_compare_cpu'))), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.celu', torch_opinfo_name='nn.functional.celu', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.threshold', torch_opinfo_name='nn.functional.threshold', supports_out=True), PythonRefInfo('_refs.nn.functional.dropout', torch_opinfo_name='nn.functional.dropout', decorators=(DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Expected: dropout is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.elu', torch_opinfo_name='nn.functional.elu', supports_out=True, decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.0012), torch.bfloat16: tol(atol=0.001, rtol=0.0012)}), 'TestUnaryUfuncs', device_type='cuda')]), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.hardtanh', torch_opinfo_name='nn.functional.hardtanh', supports_out=True), PythonRefInfo('_refs.nn.functional.gelu', torch_opinfo_name='nn.functional.gelu'), PythonRefInfo('_refs.nn.functional.layer_norm', torch_opinfo_name='nn.functional.layer_norm', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref', dtypes=(torch.float32,), device_type='cpu'),)), PythonRefInfo('_refs.nn.functional.glu', torch_opinfo_name='nn.functional.glu', supports_out=True), PythonRefInfo('_refs.nn.functional.pairwise_distance', torch_opinfo_name='nn.functional.pairwise_distance', supports_out=True), PythonRefInfo('_refs.nn.functional.pdist', torch_opinfo_name='nn.functional.pdist', supports_out=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'),)), PythonRefInfo('_refs.nn.functional.leaky_relu', torch_opinfo_name='nn.functional.leaky_relu', supports_out=True), PythonRefInfo('_refs.nn.functional.log_softmax', torch_opinfo_name='log_softmax', torch_opinfo_variant_name='with_dtype', supports_out=False), PythonRefInfo('_refs.nn.functional.poisson_nll_loss', torch_opinfo_name='nn.functional.poisson_nll_loss'), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.prelu', torch_opinfo_name='nn.functional.prelu'), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.relu', torch_opinfo_name='nn.functional.relu', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.relu6', torch_opinfo_name='nn.functional.relu6', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.mish', torch_opinfo_name='nn.functional.mish', supports_out=True, decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.001)}), 'TestUnaryUfuncs')]), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.selu', torch_opinfo_name='nn.functional.selu', supports_out=True, decorators=[DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0.018), torch.bfloat16: tol(atol=0.01, rtol=0.018)}), 'TestUnaryUfuncs', device_type='cuda')]), PythonRefInfo('_refs.nn.functional.softmax', torch_opinfo_name='softmax', torch_opinfo_variant_name='with_dtype', supports_out=False), PythonRefInfo('_refs.nn.functional.softmin', torch_opinfo_name='nn.functional.softmin', torch_opinfo_variant_name='with_dtype', supports_out=False), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.softplus', torch_opinfo_name='nn.functional.softplus'), PythonRefInfo('_refs.nn.functional.l1_loss', torch_opinfo_name='nn.functional.l1_loss'), PythonRefInfo('_refs.nn.functional.margin_ranking_loss', torch_opinfo_name='nn.functional.margin_ranking_loss'), PythonRefInfo('_refs.nn.functional.mse_loss', torch_opinfo_name='nn.functional.mse_loss'), PythonRefInfo('_refs.nn.functional.smooth_l1_loss', torch_opinfo_name='nn.functional.smooth_l1_loss'), PythonRefInfo('_refs.nn.functional.hinge_embedding_loss', torch_opinfo_name='nn.functional.hinge_embedding_loss'), PythonRefInfo('_refs.nn.functional.nll_loss', torch_opinfo_name='nn.functional.nll_loss', supports_out=True, validate_view_consistency=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', device_type='cuda'),)), PythonRefInfo('_refs.nn.functional.huber_loss', torch_opinfo_name='nn.functional.huber_loss', supports_out=True), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.tanhshrink', torch_opinfo_name='nn.functional.tanhshrink', decorators=[DecorateInfo(unittest.skip('Skipped!'), 'TestUnaryUfuncs', 'test_reference_numerics_normal', device_type='cpu', dtypes=[torch.cfloat, torch.cdouble]), DecorateInfo(toleranceOverride({torch.bfloat16: tol(atol=0.01, rtol=0.016), torch.complex64: tol(atol=0.0006, rtol=1e-05)}), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', device_type='cuda')], skips=(DecorateInfo(unittest.skip('Fails on some jobs works on others!'), 'TestUnaryUfuncs', 'test_reference_numerics_large', dtypes=(torch.complex64, torch.complex128), active_if=IS_MACOS), DecorateInfo(unittest.skip('Fails on some jobs works on others!'), 'TestUnaryUfuncs', 'test_reference_numerics_extremal', dtypes=(torch.complex64, torch.complex128), device_type='cpu', active_if=IS_MACOS or IS_WINDOWS))), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.hardshrink', torch_opinfo_name='nn.functional.hardshrink'), ElementwiseUnaryPythonRefInfo('_refs.nn.functional.softshrink', torch_opinfo_name='nn.functional.softshrink'), ElementwiseBinaryPythonRefInfo('_refs.add', torch_opinfo_name='add', supports_two_python_scalars=True, supports_one_python_scalar=True, decorators=(DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.01, rtol=0)}), 'TestBinaryUfuncs', 'test_reference_numerics'),), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=(torch.complex64, torch.complex128)),)), ElementwiseBinaryPythonRefInfo('_refs.atan2', torch_opinfo_name='atan2'), ElementwiseBinaryPythonRefInfo('_refs.bitwise_and', torch_opinfo_name='bitwise_and'), ElementwiseBinaryPythonRefInfo('_refs.bitwise_left_shift', torch_opinfo_name='bitwise_left_shift', skips=(DecorateInfo(unittest.skip('Some inputs produce undefined outputs'), 'TestCommon', 'test_compare_cpu'),)), ElementwiseBinaryPythonRefInfo('_refs.bitwise_right_shift', torch_opinfo_name='bitwise_right_shift', skips=(DecorateInfo(unittest.skip('Skipped some inputs produce undefined outputs'), 'TestCommon', 'test_compare_cpu'),)), ElementwiseBinaryPythonRefInfo('_refs.bitwise_or', torch_opinfo_name='bitwise_or'), ElementwiseBinaryPythonRefInfo('_refs.bitwise_xor', torch_opinfo_name='bitwise_xor'), ElementwiseBinaryPythonRefInfo('_refs.copysign', torch_opinfo_name='copysign', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_type_promotion'),)), ElementwiseBinaryPythonRefInfo('_refs.div', torch_opinfo_name='div', torch_opinfo_variant_name='no_rounding_mode', supports_two_python_scalars=True, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref_executor', dtypes=(torch.complex32, torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.complex32,), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.complex32,), device_type='cuda'))), ElementwiseBinaryPythonRefInfo('_refs.div', torch_opinfo_name='div', torch_opinfo_variant_name='trunc_rounding', supports_two_python_scalars=True, supports_one_python_scalar=True, decorators=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'),)), ElementwiseBinaryPythonRefInfo('_refs.div', torch_opinfo_name='div', torch_opinfo_variant_name='floor_rounding', supports_two_python_scalars=True, supports_one_python_scalar=True, decorators=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'),)), ElementwiseBinaryPythonRefInfo('_refs.eq', torch_opinfo_name='eq'), ElementwiseBinaryPythonRefInfo('_refs.float_power', torch_opinfo_name='float_power', skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=[torch.complex64, torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_large_values', dtypes=[torch.complex64, torch.complex128]), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=[torch.complex64, torch.complex128]))), ElementwiseBinaryPythonRefInfo('_refs.logaddexp', torch_opinfo_name='logaddexp', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', device_type='cpu', dtypes=(torch.complex64, torch.complex128)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', device_type='cpu', dtypes=(torch.complex64, torch.complex128)))), PythonRefInfo('_refs.logaddexp2', torch_opinfo_name='logaddexp2'), ElementwiseBinaryPythonRefInfo('_refs.floor_divide', torch_opinfo_name='floor_divide', rhs_make_tensor_kwargs=dict(exclude_zero=True), supports_two_python_scalars=True, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.int8,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=(torch.float16,)), DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.001, rtol=0.005)}), 'TestBinaryUfuncs', 'test_reference_numerics'))), ElementwiseBinaryPythonRefInfo('_refs.fmax', torch_opinfo_name='fmax', supports_rhs_python_scalar=False), ElementwiseBinaryPythonRefInfo('_refs.fmin', torch_opinfo_name='fmin', supports_rhs_python_scalar=False), ElementwiseBinaryPythonRefInfo('_refs.fmod', torch_opinfo_name='fmod', rhs_make_tensor_kwargs={'exclude_zero': True}, supports_rhs_python_scalar=True, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref', dtypes=(torch.bfloat16,), device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.bfloat16,), device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_contig_vs_every_other', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_non_contig', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.uint8,)))), ElementwiseBinaryPythonRefInfo('_refs.gcd', torch_opinfo_name='gcd', skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.int8,)),)), ElementwiseBinaryPythonRefInfo('_refs.ge', torch_opinfo_name='ge'), ElementwiseBinaryPythonRefInfo('_refs.gt', torch_opinfo_name='gt'), ElementwiseBinaryPythonRefInfo('_refs.heaviside', torch_opinfo_name='heaviside', supports_rhs_python_scalar=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values'),)), ElementwiseBinaryPythonRefInfo('_refs.hypot', torch_opinfo_name='hypot', supports_rhs_python_scalar=False), ElementwiseBinaryPythonRefInfo('_refs.igamma', torch_opinfo_name='igamma'), ElementwiseBinaryPythonRefInfo('_refs.igammac', torch_opinfo_name='igammac'), ElementwiseBinaryPythonRefInfo('_refs.isclose', torch_opinfo_name='isclose', skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values'))), ElementwiseBinaryPythonRefInfo('_refs.lcm', torch_opinfo_name='lcm'), ElementwiseBinaryPythonRefInfo('_refs.le', torch_opinfo_name='le'), ElementwiseBinaryPythonRefInfo('_refs.logical_and', torch_opinfo_name='logical_and'), ElementwiseUnaryPythonRefInfo('_refs.logical_not', torch_opinfo_name='logical_not'), ElementwiseBinaryPythonRefInfo('_refs.logical_or', torch_opinfo_name='logical_or'), ElementwiseBinaryPythonRefInfo('_refs.logical_xor', torch_opinfo_name='logical_xor'), ElementwiseBinaryPythonRefInfo('_refs.lt', torch_opinfo_name='lt'), ElementwiseBinaryPythonRefInfo('_refs.maximum', torch_opinfo_name='maximum', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), ElementwiseBinaryPythonRefInfo('_refs.minimum', torch_opinfo_name='minimum', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), ElementwiseBinaryPythonRefInfo('_refs.mul', torch_opinfo_name='mul', supports_two_python_scalars=True, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', dtypes=(torch.complex32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.complex32,), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.complex32,), device_type='cuda'))), ElementwiseBinaryPythonRefInfo('_refs.ne', torch_opinfo_name='ne'), ElementwiseBinaryPythonRefInfo('_refs.nextafter', torch_opinfo_name='nextafter'), ElementwiseBinaryPythonRefInfo('_refs.pow', torch_opinfo_name='pow', decorators=(DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=1.3e-05)}), 'TestBinaryUfuncs', 'test_reference_numerics'), DecorateInfo(toleranceOverride({torch.complex64: tol(atol=0.0001, rtol=1.3e-05), torch.complex128: tol(atol=0.0001, rtol=1.3e-05)}), 'TestBinaryUfuncs', 'test_scalar_support')), skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', dtypes=(torch.complex32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.complex32,), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.complex32,), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=[torch.int8, torch.int16, torch.int32, torch.int64]), DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_reference_numerics_large_values', dtypes=[torch.int16, torch.int32, torch.int64]), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.complex32,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.complex32, torch.complex64, torch.complex128)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_large_values', dtypes=(torch.complex32, torch.complex64, torch.complex128)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_extremal_values', dtypes=(torch.complex32, torch.complex64, torch.complex128)))), ElementwiseBinaryPythonRefInfo('_refs.remainder', torch_opinfo_name='remainder', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref', dtypes=(torch.bfloat16,), device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.bfloat16,), device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.bfloat16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.uint8,)))), ElementwiseBinaryPythonRefInfo('_refs.rsub', torch_opinfo_name='rsub', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.chalf,), device_type='cpu'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.chalf,), device_type='cpu'))), ElementwiseBinaryPythonRefInfo('_refs.sub', torch_opinfo_name='sub', supports_two_python_scalars=True, supports_one_python_scalar=True, decorators=(DecorateInfo(toleranceOverride({torch.float16: tol(atol=0.01, rtol=0), torch.bfloat16: tol(atol=1e-05, rtol=0.005), torch.complex32: tol(atol=1e-05, rtol=0.001)}), 'TestBinaryUfuncs', 'test_reference_numerics'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.01, rtol=0)}), 'TestCommon', 'test_complex_half_reference_testing', device_type='cpu'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.005, rtol=0)}), 'TestDecomp', 'test_comprehensive', device_type='cpu'), DecorateInfo(toleranceOverride({torch.chalf: tol(atol=0.005, rtol=0)}), 'TestDecomp', 'test_quick', device_type='cpu')), skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics', dtypes=(torch.uint8,)), DecorateInfo(unittest.skip('Skipped!'), 'TestBinaryUfuncs', 'test_reference_numerics_small_values', dtypes=(torch.uint8,)))), ElementwiseBinaryPythonRefInfo('_refs.true_divide', torch_opinfo_name='true_divide', supports_two_python_scalars=True, supports_one_python_scalar=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_executor', dtypes=(torch.complex32,)), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=(torch.complex32,), device_type='cuda'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.complex32,), device_type='cuda'))), PythonRefInfo('_refs.addcdiv', torch_opinfo_name='addcdiv'), PythonRefInfo('_refs.addcmul', torch_opinfo_name='addcmul', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref', dtypes=(torch.float16,), device_type='cpu'), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref_torch_fallback', dtypes=(torch.float16,), device_type='cpu'))), ElementwiseBinaryPythonRefInfo('_refs.clamp_min', torch_opinfo_name='clamp_min', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), ElementwiseBinaryPythonRefInfo('_refs.clamp_max', torch_opinfo_name='clamp_max', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), PythonRefInfo('_refs.clamp', torch_opinfo_name='clamp'), PythonRefInfo('_refs.nn.functional.triplet_margin_loss', torch_opinfo_name='nn.functional.triplet_margin_loss', supports_out=False, skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref', dtypes=(torch.uint8,), device_type='cpu'),)), ElementwiseBinaryPythonRefInfo('_refs.xlogy', torch_opinfo_name='xlogy', supports_one_python_scalar=True), ElementwiseBinaryPythonRefInfo('_refs.special.xlog1py', torch_opinfo_name='special.xlog1py', supports_one_python_scalar=True), ElementwiseUnaryPythonRefInfo('_refs._conversions.bfloat16', torch_opinfo_name='bfloat16', validate_view_consistency=False), ElementwiseUnaryPythonRefInfo('_refs._conversions.bool', torch_opinfo_name='bool', validate_view_consistency=False), ElementwiseUnaryPythonRefInfo('_refs._conversions.byte', torch_opinfo_name='byte', validate_view_consistency=False, skips=(DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'),)), ElementwiseUnaryPythonRefInfo('_refs._conversions.char', torch_opinfo_name='char', validate_view_consistency=False, skips=(DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'),)), ElementwiseBinaryPythonRefInfo('_refs._conversions.complex', torch_opinfo_name='complex', error_inputs_func=partial(error_inputs_complex, is_ref=True), skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'),)), ElementwiseBinaryPythonRefInfo('_refs._conversions.polar', torch_opinfo_name='polar', skips=(DecorateInfo(unittest.expectedFailure, 'TestBinaryUfuncs', 'test_type_promotion'),)), ElementwiseUnaryPythonRefInfo('_refs._conversions.double', torch_opinfo_name='double', validate_view_consistency=False), ElementwiseUnaryPythonRefInfo('_refs._conversions.float', torch_opinfo_name='float', validate_view_consistency=False), ElementwiseUnaryPythonRefInfo('_refs._conversions.half', torch_opinfo_name='half', validate_view_consistency=False), ElementwiseUnaryPythonRefInfo('_refs._conversions.int', torch_opinfo_name='int', validate_view_consistency=False, skips=(DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'),)), ElementwiseUnaryPythonRefInfo('_refs._conversions.long', torch_opinfo_name='long', validate_view_consistency=False, skips=(DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'),)), ElementwiseUnaryPythonRefInfo('_refs._conversions.short', torch_opinfo_name='short', validate_view_consistency=False, skips=(DecorateInfo(unittest.skip('Overflow when downcasting signed type is undefined'), 'TestCommon', 'test_compare_cpu'),)), ElementwiseUnaryPythonRefInfo('_refs._conversions.chalf', torch_opinfo_name='chalf', validate_view_consistency=False), ElementwiseUnaryPythonRefInfo('_refs._conversions.cfloat', torch_opinfo_name='cfloat', validate_view_consistency=False), ElementwiseUnaryPythonRefInfo('_refs._conversions.cdouble', torch_opinfo_name='cdouble', validate_view_consistency=False), PythonRefInfo('_refs.clone', torch_opinfo_name='clone'), PythonRefInfo('_refs.atleast_1d', torch_opinfo_name='atleast_1d', validate_view_consistency=False), PythonRefInfo('_refs.atleast_2d', torch_opinfo_name='atleast_2d', validate_view_consistency=False), PythonRefInfo('_refs.atleast_3d', torch_opinfo_name='atleast_3d', validate_view_consistency=False), PythonRefInfo('_refs.as_strided', torch_opinfo_name='as_strided', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.as_strided', torch_opinfo_name='as_strided', torch_opinfo_variant_name='partial_views', dtypes=all_types_and_complex_and(torch.bool, torch.float16, torch.bfloat16), skips=(DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Errors when storage_offset is included'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_compare_cpu'))), PythonRefInfo('_refs.as_strided_scatter', torch_opinfo_name='as_strided_scatter', validate_view_consistency=False), PythonRefInfo('_refs.broadcast_shapes', torch_opinfo_name='broadcast_shapes'), PythonRefInfo('_refs.broadcast_tensors', torch_opinfo_name='broadcast_tensors'), PythonRefInfo('_refs.broadcast_to', torch_opinfo_name='broadcast_to'), PythonRefInfo('_refs.cat', torch_opinfo_name='cat', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), PythonRefInfo('_refs.chunk', torch_opinfo_name='chunk'), PythonRefInfo('_refs.column_stack', torch_opinfo_name='column_stack'), ElementwiseUnaryPythonRefInfo('_refs.conj', torch_opinfo_name='conj'), PythonRefInfo('_refs.constant_pad_nd', torch_opinfo_name='constant_pad_nd'), PythonRefInfo('_refs.contiguous', torch_opinfo_name='contiguous'), ElementwiseUnaryPythonRefInfo('_refs.deg2rad', torch_opinfo_name='deg2rad', decorators=(precisionOverride({torch.bfloat16: 0.7, torch.float16: 0.7}),)), PythonRefInfo('_refs.dsplit', torch_opinfo_name='dsplit'), PythonRefInfo('_refs.diag', torch_opinfo_name='diag'), PythonRefInfo('_refs.diagonal', torch_opinfo_name='diagonal'), PythonRefInfo('_refs.diagonal_copy', torch_opinfo_name='diagonal_copy'), PythonRefInfo('_refs.diagonal_scatter', torch_opinfo_name='diagonal_scatter', supports_out=True, validate_view_consistency=False), PythonRefInfo('_refs.diag_embed', torch_opinfo_name='diag_embed', supports_out=True), PythonRefInfo('_refs.dstack', torch_opinfo_name='dstack', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), PythonRefInfo('_refs.expand', torch_opinfo_name='expand'), PythonRefInfo('_refs.expand_as', torch_opinfo_name='expand_as'), PythonRefInfo('_refs.flatten', torch_opinfo_name='flatten'), PythonRefInfo('_refs.flip', torch_opinfo_name='flip'), PythonRefInfo('_refs.fliplr', torch_opinfo_name='fliplr'), PythonRefInfo('_refs.flipud', torch_opinfo_name='flipud'), PythonRefInfo('_refs.hstack', torch_opinfo_name='hstack', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), PythonRefInfo('_refs.narrow', torch_opinfo_name='narrow', error_inputs_func=partial(error_inputs_narrow_narrow_copy, is_narrow=True, is_ref=True)), PythonRefInfo('_refs.narrow_copy', torch_opinfo_name='narrow_copy', supports_out=True, error_inputs_func=partial(error_inputs_narrow_narrow_copy, is_narrow=False, is_ref=True)), PythonRefInfo('_refs.nn.functional.group_norm', torch_opinfo_name='nn.functional.group_norm', validate_view_consistency=False), PythonRefInfo('_refs.native_layer_norm', torch_opinfo_name='native_layer_norm', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref', device_type='cpu', dtypes=(torch.float32,)), DecorateInfo(unittest.skip('Skipped!'), 'TestCommon', 'test_python_ref_torch_fallback', device_type='cpu', dtypes=(torch.float32,)))), PythonRefInfo('_refs.permute', torch_opinfo_name='permute'), ElementwiseUnaryPythonRefInfo('_refs.rad2deg', torch_opinfo_name='rad2deg', decorators=(precisionOverride({torch.bfloat16: 0.7, torch.float16: 0.7}),)), PythonRefInfo('_refs.ravel', torch_opinfo_name='ravel'), PythonRefInfo('_refs.renorm', torch_opinfo_name='renorm'), PythonRefInfo('_refs.repeat', torch_opinfo_name='repeat', validate_view_consistency=False), PythonRefInfo('_refs.reshape', torch_opinfo_name='reshape'), PythonRefInfo('_refs.reshape_as', torch_opinfo_name='reshape_as'), PythonRefInfo('_refs.roll', torch_opinfo_name='roll', validate_view_consistency=False), PythonRefInfo('_refs.rot90', torch_opinfo_name='rot90', validate_view_consistency=False), PythonRefInfo('_refs.stack', torch_opinfo_name='stack', validate_view_consistency=False), PythonRefInfo('_refs.squeeze', torch_opinfo_name='squeeze'), PythonRefInfo('_refs.squeeze', torch_opinfo_name='squeeze', torch_opinfo_variant_name='multiple'), PythonRefInfo('_refs.tensor_split', torch_opinfo_name='tensor_split', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_meta'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'))), PythonRefInfo('_refs.hsplit', torch_opinfo_name='hsplit'), PythonRefInfo('_refs.vsplit', torch_opinfo_name='vsplit'), PythonRefInfo('_refs.dot', torch_opinfo_name='dot', error_inputs_func=partial(error_inputs_dot_vdot, is_ref=True), validate_view_consistency=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=[torch.complex64, torch.complex128]),)), PythonRefInfo('_refs.vdot', torch_opinfo_name='vdot', error_inputs_func=partial(error_inputs_dot_vdot, is_ref=True), validate_view_consistency=False, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref', dtypes=[torch.complex64, torch.complex128]),)), PythonRefInfo('_refs.transpose', torch_opinfo_name='transpose'), PythonRefInfo('_refs.t', torch_opinfo_name='t'), PythonRefInfo('_refs.T', torch_opinfo_name='T', error_inputs_func=partial(error_inputs_T, has_ndims_error=True)), PythonRefInfo('_refs.unfold', torch_opinfo_name='unfold'), PythonRefInfo('_refs.unfold_copy', torch_opinfo_name='unfold_copy', supports_out=True), PythonRefInfo('_refs.unsqueeze', torch_opinfo_name='unsqueeze'), PythonRefInfo('_refs.view', torch_opinfo_name='view'), PythonRefInfo('_refs.view_as', torch_opinfo_name='view_as'), PythonRefInfo('_refs.vstack', torch_opinfo_name='vstack', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), PythonRefInfo('_refs.unflatten', torch_opinfo_name='unflatten'), PythonRefInfo('_refs.unbind', torch_opinfo_name='unbind'), ReductionPythonRefInfo('_refs.all', torch_opinfo_name='all', skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_result_dtype', dtypes=[torch.uint8]),)), ReductionPythonRefInfo('_refs.amax', torch_opinfo_name='amax', error_inputs_func=partial(error_inputs_aminmax_amax_amin, is_ref=True), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'))), ReductionPythonRefInfo('_refs.amin', torch_opinfo_name='amin', error_inputs_func=partial(error_inputs_aminmax_amax_amin, is_ref=True), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'))), ReductionPythonRefInfo('_refs.any', torch_opinfo_name='any', skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_result_dtype', dtypes=[torch.uint8]),)), ReductionPythonRefInfo('_refs.count_nonzero', torch_opinfo_name='count_nonzero', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_default_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_none_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_single_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_multi_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_multi_unsorted_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'))), ReductionPythonRefInfo('_refs.mean', torch_opinfo_name='mean', supports_out=True, error_inputs_func=partial(error_inputs_mean, is_ref=True), skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'))), ReductionPythonRefInfo('_refs.std', torch_opinfo_name='std', supports_out=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=(torch.float16,)), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_duplicate_values', dtypes=(torch.float16,)))), PythonRefInfo('_refs.std_mean', torch_opinfo_name='std_mean'), ReductionPythonRefInfo('_refs.sum', torch_opinfo_name='sum', supports_out=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=[torch.float16]), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_duplicate_values', dtypes=[torch.float16]), DecorateInfo(unittest.skip('Skipped!'), 'TestOperators', 'test_reduction_all', dtypes=[torch.float32]))), PythonRefInfo('_refs.cumsum', torch_opinfo_name='cumsum', supports_out=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),)), PythonRefInfo('_refs.cumprod', torch_opinfo_name='cumprod', supports_out=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'),)), PythonRefInfo('_refs.sum_to_size', torch_opinfo_name='sum_to_size', validate_view_consistency=False), ReductionPythonRefInfo('_refs.prod', torch_opinfo_name='prod', supports_out=True, supports_multiple_dims=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_out'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input', dtypes=[torch.float16, torch.complex64]))), ReductionPythonRefInfo('_refs.var', torch_opinfo_name='var', supports_out=True, skips=(DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty'), DecorateInfo(unittest.expectedFailure, 'TestReductions', 'test_dim_empty_keepdim'), DecorateInfo(unittest.skip('Skipped!'), 'TestReductions', 'test_ref_small_input'))), PythonRefInfo('_refs.var_mean', torch_opinfo_name='var_mean', validate_view_consistency=False), PythonRefInfo('_refs.addr', torch_opinfo_name='addr', decorators=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'),)), PythonRefInfo('_refs.trace', torch_opinfo_name='trace'), PythonRefInfo('_refs.norm', torch_opinfo_name='norm', supports_out=True, validate_view_consistency=False), PythonRefInfo('_refs.empty', torch_opinfo_name='empty', skips=(DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip("Can't check result for empty"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), PythonRefInfo('_refs.empty_like', torch_opinfo_name='empty_like', skips=(DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip("Can't check result for empty_like"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), PythonRefInfo('_refs.randn', torch_opinfo_name='randn', op=lambda *args, **kwargs: wrapper_set_seed(refs.randn, *args, **kwargs), skips=(DecorateInfo(unittest.skip("make_traced() doesn't set seed properly!"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestCommon', 'test_noncontiguous_samples'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Test expects tensor input'), 'TestMathBits', 'test_neg_conj_view'))), PythonRefInfo('_refs.eye', torch_opinfo_name='eye', skips=(DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Skipped!'), 'TestMathBits', 'test_neg_view'))), PythonRefInfo('_refs.new_empty', torch_opinfo_name='new_empty', skips=(DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestCommon', 'test_out_warning'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Expected: empty is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip("Can't check result for new_empty"), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), PythonRefInfo('_refs.new_empty_strided', torch_opinfo_name='new_empty_strided', skips=(DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), PythonRefInfo('_refs.empty_strided', torch_opinfo_name='empty_strided', skips=(DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestCommon', 'test_python_ref_torch_fallback'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestMathBits', 'test_conj_view'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestMathBits', 'test_neg_conj_view'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestMathBits', 'test_neg_view'), DecorateInfo(unittest.skip('Expected: empty_strided is not comparable'), 'TestCommon', 'test_python_ref_executor'), DecorateInfo(unittest.skip('output is non-deterministic'), 'TestCommon', 'test_compare_cpu'))), PythonRefInfo('_refs.new_full', torch_opinfo_name='new_full'), PythonRefInfo('_refs.new_ones', torch_opinfo_name='new_ones'), PythonRefInfo('_refs.new_zeros', torch_opinfo_name='new_zeros'), PythonRefInfo('_refs.masked_fill', torch_opinfo_name='masked_fill', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'),)), PythonRefInfo('_refs.where', torch_opinfo_name='where', op=lambda self, condition, other: refs.where(condition, self, other), skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors', device_type='cuda'),)), PythonRefInfo('_refs.index_select', torch_opinfo_name='index_select', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'))), PythonRefInfo('_refs.index_copy', torch_opinfo_name='index_copy', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'),)), PythonRefInfo('_refs.index_add', torch_opinfo_name='index_add', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'), DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref_errors'))), PythonRefInfo('_refs.index_fill', torch_opinfo_name='index_fill', skips=(DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref'),)), PythonRefInfo('_refs.allclose', torch_opinfo_name='allclose'), PythonRefInfo('_refs.stft', torch_opinfo_name='stft', skips=[DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref')]), PythonRefInfo('_refs.istft', torch_opinfo_name='istft', skips=[DecorateInfo(unittest.expectedFailure, 'TestCommon', 'test_python_ref')]), PythonRefInfo('_refs.view_as_complex', torch_opinfo_name='view_as_complex')]
python_ref_db += opinfo.definitions.python_ref_db
ops_and_refs = op_db + python_ref_db
unary_ufuncs = [op for op in ops_and_refs if isinstance(op, UnaryUfuncInfo)]
binary_ufuncs = [op for op in ops_and_refs if isinstance(op, BinaryUfuncInfo)]
binary_ufuncs_and_refs = tuple((op for op in ops_and_refs if isinstance(op, BinaryUfuncInfo)))
spectral_funcs = [op for op in ops_and_refs if isinstance(op, SpectralFuncInfo)]
sparse_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse]
sparse_csr_unary_ufuncs = [op for op in op_db if isinstance(op, UnaryUfuncInfo) and op.supports_sparse_csr]
sparse_reduction_ops = [op for op in op_db if isinstance(op, ReductionOpInfo) and op.supports_sparse]
shape_funcs = [op for op in ops_and_refs if isinstance(op, ShapeFuncInfo)]
reduction_ops = [op for op in ops_and_refs if isinstance(op, ReductionOpInfo)]
reference_filtered_ops = [op for op in reduction_ops if op.ref is not None]
reference_masked_ops = [op for op in reference_filtered_ops if op.name.startswith('masked.')]
sparse_masked_reduction_ops = [op for op in sparse_reduction_ops if op.name.startswith('masked.')]

def index_variable(shape, max_indices, device=torch.device('cpu')):
    if False:
        while True:
            i = 10
    if not isinstance(shape, tuple):
        shape = (shape,)
    index = torch.rand(*shape, dtype=torch.double, device=device).mul_(max_indices).floor_().long()
    return index

def gather_variable(shape, index_dim, max_indices, duplicate=False, device=torch.device('cpu')):
    if False:
        print('Hello World!')
    assert len(shape) == 2
    assert index_dim < 2
    batch_dim = 1 - index_dim
    index = torch.zeros(*shape, dtype=torch.long, device=device)
    for i in range(shape[index_dim]):
        index.select(index_dim, i).copy_(torch.randperm(max_indices, device=device)[:shape[batch_dim]])
    if duplicate:
        index.select(batch_dim, 0).copy_(index.select(batch_dim, 1))
    return index

def bernoulli_scalar():
    if False:
        for i in range(10):
            print('nop')
    return torch.tensor(0, dtype=torch.bool).bernoulli_()

def mask_not_all_zeros(shape):
    if False:
        print('Hello World!')
    assert len(shape) > 0
    while True:
        result = torch.randn(shape).gt(0)
        if result.sum() > 0:
            return result

def xfail(op_name, variant_name='', *, device_type=None, dtypes=None):
    if False:
        while True:
            i = 10
    return (op_name, variant_name, device_type, dtypes, True)

def skip(op_name, variant_name='', *, device_type=None, dtypes=None):
    if False:
        while True:
            i = 10
    return (op_name, variant_name, device_type, dtypes, False)

def skipOps(test_case_name, base_test_name, to_skip):
    if False:
        print('Hello World!')
    all_opinfos = op_db
    for xfail in to_skip:
        (op_name, variant_name, device_type, dtypes, expected_failure) = xfail
        matching_opinfos = [o for o in all_opinfos if o.name == op_name and o.variant_test_name == variant_name]
        assert len(matching_opinfos) >= 1, f"Couldn't find OpInfo for {xfail}"
        for op in matching_opinfos:
            decorators = list(op.decorators)
            if expected_failure:
                decorator = DecorateInfo(unittest.expectedFailure, test_case_name, base_test_name, device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            else:
                decorator = DecorateInfo(unittest.skip('Skipped!'), test_case_name, base_test_name, device_type=device_type, dtypes=dtypes)
                decorators.append(decorator)
            op.decorators = tuple(decorators)

    def wrapped(fn):
        if False:
            for i in range(10):
                print('nop')
        return fn
    return wrapped