import collections
import functools
import warnings
from itertools import product
from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union
import torch
import torch.testing
from torch._vmap_internals import _vmap, vmap
from torch.overrides import is_tensor_like
from torch.types import _TensorOrTensors
__all__ = ['gradcheck', 'gradgradcheck', 'GradcheckError', 'get_numerical_jacobian', 'get_analytical_jacobian', 'get_numerical_jacobian_wrt_specific_input']

class GradcheckError(RuntimeError):
    """Error raised by :func:`gradcheck` and :func:`gradgradcheck`."""
    pass

def _is_sparse_compressed_tensor(obj: torch.Tensor):
    if False:
        while True:
            i = 10
    return obj.layout in {torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc}

def _is_sparse_any_tensor(obj: torch.Tensor):
    if False:
        print('Hello World!')
    return _is_sparse_compressed_tensor(obj) or obj.layout is torch.sparse_coo

def _is_float_or_complex_tensor(obj):
    if False:
        while True:
            i = 10
    return is_tensor_like(obj) and (obj.is_floating_point() or obj.is_complex())

def _allocate_jacobians_with_inputs(input_tensors: Tuple, numel_output) -> Tuple[torch.Tensor, ...]:
    if False:
        return 10
    out: List[torch.Tensor] = []
    for t in input_tensors:
        if _is_float_or_complex_tensor(t) and t.requires_grad:
            out.append(t.new_zeros((t.numel(), numel_output), layout=torch.strided))
    return tuple(out)

def _allocate_jacobians_with_outputs(output_tensors: Tuple, numel_input, dtype=None, device=None) -> Tuple[torch.Tensor, ...]:
    if False:
        return 10
    out: List[torch.Tensor] = []
    options = {'dtype': dtype, 'device': device, 'layout': torch.strided}
    for t in output_tensors:
        if _is_float_or_complex_tensor(t):
            out.append(t.new_zeros((numel_input, t.numel()), **options))
    return tuple(out)

def _iter_tensors(x: Union[torch.Tensor, Iterable[torch.Tensor]], only_requiring_grad: bool=False) -> Iterable[torch.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    if is_tensor_like(x):
        if x.requires_grad or not only_requiring_grad:
            yield x
    elif isinstance(x, collections.abc.Iterable) and (not isinstance(x, str)):
        for elem in x:
            yield from _iter_tensors(elem, only_requiring_grad)

def _densify(x):
    if False:
        i = 10
        return i + 15
    if isinstance(x, (list, tuple)):
        return type(x)(map(_densify, x))
    elif not is_tensor_like(x) or x.layout in {torch.strided, torch._mkldnn}:
        return x
    elif x.layout is torch.sparse_coo:
        device = x.device
        indices_dtype = x._indices().dtype
        tmp = torch.ones(x.shape[:x.sparse_dim()], dtype=torch.int8, device=device)
        indices = tmp.nonzero().t().to(dtype=indices_dtype)
        values = torch.zeros((tmp.numel(), *x.shape[x.sparse_dim():]), dtype=x.dtype, device=device)
        x_coalesced = x.detach().coalesce()
        if x_coalesced.numel() > 0:
            stride = tmp.stride()
            flat_indices = x_coalesced.indices().mul(torch.tensor(stride, dtype=indices_dtype, device=device).unsqueeze(1)).sum(0)
            values[flat_indices] = x_coalesced.values()
        return torch.sparse_coo_tensor(indices, values, x.shape)._coalesced_(True).requires_grad_(x.requires_grad)
    elif _is_sparse_compressed_tensor(x):
        blocksize = x.values().shape[1:3] if x.layout in {torch.sparse_bsr, torch.sparse_bsc} else None
        compressed_indices = x.crow_indices() if x.layout in {torch.sparse_csr, torch.sparse_bsr} else x.ccol_indices()
        r = _densify(x.detach().to_sparse(layout=torch.sparse_coo)).to_sparse(layout=x.layout, blocksize=blocksize)
        dense_numel = r.values().numel() // max(1, r.values().shape[0])
        batch_numel = compressed_indices.numel() // compressed_indices.shape[-1]
        sparse_numel = r.numel() // max(1, dense_numel * batch_numel)
        if sparse_numel != r._nnz():
            raise AssertionError(f'{x.layout} densify failed: expected nnz={sparse_numel} but got {r._nnz()}')
        return r.requires_grad_(x.requires_grad)
    elif _is_sparse_any_tensor(x):
        raise NotImplementedError(x.layout)
    return x

def _iter_tensor(x_tensor):
    if False:
        return 10
    if _is_sparse_any_tensor(x_tensor):

        def get_stride(size):
            if False:
                return 10
            dim = len(size)
            tmp = 1
            stride = [0] * dim
            for i in reversed(range(dim)):
                stride[i] = tmp
                tmp *= size[i]
            return stride
        x_nnz = x_tensor._nnz()
        x_size = list(x_tensor.size())
        if x_tensor.layout is torch.sparse_coo:
            x_indices = x_tensor._indices().t()
            x_values = x_tensor._values()
        elif x_tensor.layout is torch.sparse_csr:
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.crow_indices(), x_tensor.col_indices()).t()
            x_values = x_tensor.values()
        elif x_tensor.layout is torch.sparse_csc:
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.ccol_indices(), x_tensor.row_indices(), transpose=True).t()
            x_values = x_tensor.values()
        elif x_tensor.layout is torch.sparse_bsr:
            x_block_values = x_tensor.values()
            x_blocksize = x_block_values.size()[1:3]
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.crow_indices(), x_tensor.col_indices()).repeat_interleave(x_blocksize[0] * x_blocksize[1], 1).mul_(torch.tensor(x_blocksize, device=x_tensor.device).reshape(2, 1)).add_(torch.stack(torch.where(torch.ones(x_blocksize, device=x_tensor.device))).repeat(1, x_nnz)).t()
            x_values = x_block_values.flatten(0, 2)
            x_nnz = x_values.size(0)
        elif x_tensor.layout is torch.sparse_bsc:
            x_block_values = x_tensor.values()
            x_blocksize = x_block_values.size()[1:3]
            x_indices = torch._convert_indices_from_csr_to_coo(x_tensor.ccol_indices(), x_tensor.row_indices(), transpose=True).repeat_interleave(x_blocksize[0] * x_blocksize[1], 1).mul_(torch.tensor(x_blocksize, device=x_tensor.device).reshape(2, 1)).add_(torch.stack(torch.where(torch.ones(x_blocksize, device=x_tensor.device))).repeat(1, x_nnz)).t()
            x_values = x_block_values.flatten(0, 2)
            x_nnz = x_values.size(0)
        else:
            raise NotImplementedError(f'_iter_tensor for {x_tensor.layout} input')
        x_stride = get_stride(x_size)
        x_values = x_values.data
        for i in range(x_nnz):
            x_value = x_values[i]
            for x_idx in product(*[range(m) for m in x_values.size()[1:]]):
                indices = x_indices[i].tolist() + list(x_idx)
                d_idx = sum((indices[k] * x_stride[k] for k in range(len(x_size))))
                yield (x_value, x_idx, d_idx)
    elif x_tensor.layout == torch._mkldnn:
        for (d_idx, x_idx) in enumerate(product(*[range(m) for m in x_tensor.size()])):
            x_tensor_dense = x_tensor.to_dense()
            yield (x_tensor_dense, x_idx, d_idx)
    else:
        x_tensor = x_tensor.data
        for (d_idx, x_idx) in enumerate(product(*[range(m) for m in x_tensor.size()])):
            yield (x_tensor, x_idx, d_idx)

def _get_numerical_jacobian(fn, inputs, outputs=None, target=None, eps=0.001, is_forward_ad=False) -> List[Tuple[torch.Tensor, ...]]:
    if False:
        for i in range(10):
            print('nop')
    'Compute the numerical Jacobian of `fn(inputs)` with respect to `target`.\n\n    If not specified, targets are the input. Returns M * N Jacobians where N is the\n    number of tensors in target that require grad and M is the number of non-integral\n    outputs.\n\n    Args:\n        fn: the function to compute the jacobian for\n        inputs: inputs to `fn`\n        outputs: provide precomputed outputs to avoid one extra invocation of fn\n        target: the Tensors wrt whom Jacobians are calculated (default=`inputs`)\n        eps: the magnitude of the perturbation during finite differencing\n             (default=`1e-3`)\n        is_forward_ad: if this numerical jacobian is computed to be checked wrt\n                       forward AD gradients (this is used for error checking only)\n\n    Returns:\n        A list of M N-tuples of tensors\n\n    Note that `target` may not even be part of `input` to `fn`, so please be\n    **very careful** in this to not clone `target`.\n    '
    jacobians: List[Tuple[torch.Tensor, ...]] = []
    if outputs is None:
        outputs = _as_tuple(fn(*_as_tuple(inputs)))
    if not is_forward_ad and any((o.is_complex() for o in outputs)):
        raise ValueError('Expected output to be non-complex. get_numerical_jacobian no longer supports functions that return complex outputs.')
    if target is None:
        target = inputs
    inp_indices = [i for (i, a) in enumerate(target) if is_tensor_like(a) and a.requires_grad]
    for (i, (inp, inp_idx)) in enumerate(zip(_iter_tensors(target, True), inp_indices)):
        jacobians += [get_numerical_jacobian_wrt_specific_input(fn, inp_idx, inputs, outputs, eps, input=inp, is_forward_ad=is_forward_ad)]
    return jacobians

def get_numerical_jacobian(fn, inputs, target=None, eps=0.001, grad_out=1.0):
    if False:
        i = 10
        return i + 15
    'Compute the numerical Jacobian for a given fn and its inputs.\n\n    This is a Deprecated API.\n\n    Args:\n        fn: the function to compute the Jacobian for (must take inputs as a tuple)\n        input: input to `fn`\n        target: the Tensors wrt whom Jacobians are calculated (default=`input`)\n        eps: the magnitude of the perturbation during finite differencing\n             (default=`1e-3`)\n\n    Returns:\n        A list of Jacobians of `fn` (restricted to its first output) with respect to\n        each input or target, if provided.\n\n    Note that `target` may not even be part of `input` to `fn`, so please be\n    **very careful** in this to not clone `target`.\n    '
    warnings.warn("get_numerical_jacobian was part of PyTorch's private API and not meant to be exposed. We are deprecating it and it will be removed in a future version of PyTorch. If you have a specific use for this or feature request for this to be a stable API, please file us an issue at https://github.com/pytorch/pytorch/issues/new")
    if grad_out != 1.0:
        raise ValueError('Expected grad_out to be 1.0. get_numerical_jacobian no longer supports values of grad_out != 1.0.')

    def fn_pack_inps(*inps):
        if False:
            return 10
        return fn(inps)
    jacobians = _get_numerical_jacobian(fn_pack_inps, inputs, None, target, eps)
    return tuple((jacobian_for_each_output[0] for jacobian_for_each_output in jacobians))

def _compute_numerical_gradient(fn, entry, v, norm_v, nbhd_checks_fn):
    if False:
        for i in range(10):
            print('nop')
    if _is_sparse_compressed_tensor(entry):
        assert entry.layout == v.layout, (entry.layout, v.layout)
        assert entry._nnz() == v._nnz(), (entry._nnz(), v._nnz(), entry.shape)
        entry = entry.values()
        v = v.values()
        entry = entry.detach()
    orig = entry.clone()
    entry.copy_(orig - v)
    outa = fn()
    entry.copy_(orig + v)
    outb = fn()
    entry.copy_(orig)

    def compute(a, b):
        if False:
            return 10
        nbhd_checks_fn(a, b)
        ret = (b - a) / (2 * norm_v)
        return ret.detach().reshape(-1)
    return tuple((compute(a, b) for (a, b) in zip(outa, outb)))

def _compute_numerical_jvps_wrt_specific_input(jvp_fn, delta, input_is_complex, is_forward_ad=False) -> List[torch.Tensor]:
    if False:
        for i in range(10):
            print('nop')
    jvps: List[torch.Tensor] = []
    ds_dx_tup = jvp_fn(delta[0] if isinstance(delta, tuple) else delta)
    if input_is_complex:
        ds_dy_tup = jvp_fn(delta[1] * 1j) if isinstance(delta, tuple) else jvp_fn(delta * 1j)
        for (ds_dx, ds_dy) in zip(ds_dx_tup, ds_dy_tup):
            assert not ds_dx.is_complex()
            conj_w_d = ds_dx + ds_dy * 1j
            jvps.append(conj_w_d)
    else:
        for ds_dx in ds_dx_tup:
            assert is_forward_ad or not ds_dx.is_complex()
            jvps.append(ds_dx)
    return jvps

def _combine_jacobian_cols(jacobians_cols: Dict[int, List[torch.Tensor]], outputs, input, numel) -> Tuple[torch.Tensor, ...]:
    if False:
        return 10
    jacobians = _allocate_jacobians_with_outputs(outputs, numel, dtype=input.dtype if input.dtype.is_complex else None)
    for (i, jacobian) in enumerate(jacobians):
        for (k, v) in jacobians_cols.items():
            jacobian[k] = v[i]
    return jacobians

def _prepare_input(input: torch.Tensor, maybe_perturbed_input: Optional[torch.Tensor], fast_mode=False) -> torch.Tensor:
    if False:
        return 10
    if input.layout == torch._mkldnn:
        if maybe_perturbed_input is not None:
            return maybe_perturbed_input.to_mkldnn()
        else:
            return input
    elif _is_sparse_any_tensor(input):
        if fast_mode and maybe_perturbed_input is not None:
            return maybe_perturbed_input
        else:
            return input
    else:
        return input

def _check_outputs_same_dtype_and_shape(output1, output2, eps, idx=None) -> None:
    if False:
        print('Hello World!')
    on_index = 'on index {idx} ' if idx is not None else ''
    assert output1.shape == output2.shape, f'Expected `func` to return outputs with the same shape when inputs are perturbed {on_index}by {eps}, but got: shapes {output1.shape} and {output2.shape}.'
    assert output1.dtype == output2.dtype, f'Expected `func` to return outputs with the same dtype when inputs are perturbed {on_index}by {eps}, but got: dtypes {output1.dtype} and {output2.dtype}.'

def get_numerical_jacobian_wrt_specific_input(fn, input_idx, inputs, outputs, eps, input=None, is_forward_ad=False) -> Tuple[torch.Tensor, ...]:
    if False:
        while True:
            i = 10
    jacobian_cols: Dict[int, List[torch.Tensor]] = {}
    input = inputs[input_idx] if input is None else input
    assert input.requires_grad
    for (x, idx, d_idx) in _iter_tensor(input):
        wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, x)
        input_to_perturb = x[idx]
        nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, idx=idx, eps=eps)
        jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
        jacobian_cols[d_idx] = _compute_numerical_jvps_wrt_specific_input(jvp_fn, eps, x.is_complex(), is_forward_ad)
    return _combine_jacobian_cols(jacobian_cols, outputs, input, input.numel())

def _get_analytical_jacobian_forward_ad(fn, inputs, outputs, *, check_grad_dtypes=False, all_u=None) -> Tuple[Tuple[torch.Tensor, ...], ...]:
    if False:
        return 10
    'Compute the analytical Jacobian using forward mode AD of `fn(inputs)` using forward mode AD with respect to `target`.\n\n    Return N * M Jacobians where N is the number of tensors in target that require grad and\n    M is the number of non-integral outputs.\n    Contrary to other functions here, this function requires "inputs" to actually be used by the function.\n    The computed value is expected to be wrong if the function captures the inputs by side effect instead of\n    using the passed ones (many torch.nn tests do this).\n\n    Args:\n        fn: the function to compute the jacobian for\n        inputs: inputs to `fn`\n        outputs: provide precomputed outputs to avoid one extra invocation of fn\n        check_grad_dtypes: if True, will check that the gradient dtype are valid\n        all_u (optional): if provided, the Jacobian will be right multiplied with this vector\n\n    Returns:\n        A tuple of M N-tuples of tensors\n    '
    fwAD = torch.autograd.forward_ad
    tensor_inputs = tuple((i for i in inputs if is_tensor_like(i) and i.requires_grad))
    if any((i.is_complex() for i in tensor_inputs)):
        raise ValueError('Expected inputs to be non-complex for _get_analytical_jacobian_forward_ad.')
    if all_u:
        jacobians = tuple((_allocate_jacobians_with_outputs(outputs, 1) for i in tensor_inputs))
    else:
        jacobians = tuple((_allocate_jacobians_with_outputs(outputs, i.numel()) for i in tensor_inputs))
    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        for (i, inp) in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:
                    raise ValueError('MKLDNN inputs are not support for forward AD gradcheck.')
                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                fw_grads.append(fwAD.unpack_dual(inp)[1])
            dual_inputs.append(inp)
        if all_u:
            for (i, (fw_grad, u)) in enumerate(zip(fw_grads, all_u)):
                fw_grad.copy_(u.view_as(fw_grad))
                raw_outputs = _as_tuple(fn(*dual_inputs))
                dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                for (index_o, d_o) in enumerate(dual_outputs):
                    (val, res) = fwAD.unpack_dual(d_o)
                    if check_grad_dtypes and res is not None and (val.is_complex() != res.is_complex()):
                        raise GradcheckError('Forward AD gradient has dtype mismatch.')
                    jacobians[i][index_o].squeeze_(0)
                    if res is None:
                        jacobians[i][index_o].zero_()
                    else:
                        jacobians[i][index_o].copy_(res.reshape(-1))
                fw_grad.zero_()
        else:
            for (i, fw_grad) in enumerate(fw_grads):
                for (lin_idx, grad_idx) in enumerate(product(*[range(m) for m in fw_grad.size()])):
                    fw_grad[grad_idx] = 1.0
                    raw_outputs = _as_tuple(fn(*dual_inputs))
                    dual_outputs = filter(_is_float_or_complex_tensor, raw_outputs)
                    for (index_o, d_o) in enumerate(dual_outputs):
                        (val, res) = fwAD.unpack_dual(d_o)
                        if check_grad_dtypes and res is not None and (val.is_complex() != res.is_complex()):
                            raise GradcheckError('Forward AD gradient has dtype mismatch.')
                        if res is None:
                            jacobians[i][index_o][lin_idx].zero_()
                        else:
                            jacobians[i][index_o][lin_idx].copy_(res.reshape(-1))
                    fw_grad[grad_idx] = 0.0
    return jacobians

def _get_input_to_perturb(input):
    if False:
        i = 10
        return i + 15
    if input.layout == torch._mkldnn:
        input_to_perturb = input.to_dense()
    elif _is_sparse_any_tensor(input):
        input_to_perturb = input.clone()
    else:
        input_to_perturb = input.data
    return input_to_perturb

def _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, fast_mode=False):
    if False:
        return 10

    def wrapped_fn():
        if False:
            i = 10
            return i + 15
        inp = tuple((_prepare_input(a, input_to_perturb if i == input_idx else None, fast_mode) if is_tensor_like(a) else a for (i, a) in enumerate(_as_tuple(inputs))))
        return tuple((a.clone() for a in _as_tuple(fn(*inp))))
    return wrapped_fn

def _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn):
    if False:
        print('Hello World!')

    def jvp_fn(delta):
        if False:
            for i in range(10):
                print('nop')
        return _compute_numerical_gradient(wrapped_fn, input_to_perturb, delta, eps, nbhd_checks_fn)
    return jvp_fn

def _reshape_tensor_or_tuple(u, shape):
    if False:
        return 10
    if isinstance(u, tuple):
        if not _is_sparse_any_tensor(u[0]):
            return (u[0].reshape(shape), u[1].reshape(shape))
    elif not _is_sparse_any_tensor(u):
        return u.reshape(shape)
    return u

def _mul_tensor_or_tuple(u, k):
    if False:
        return 10
    if isinstance(u, tuple):
        return (k * u[0], k * u[1])
    else:
        return k * u

def _get_numerical_jvp_wrt_specific_input(fn, input_idx, inputs, u, eps, is_forward_ad=False) -> List[torch.Tensor]:
    if False:
        print('Hello World!')
    input = inputs[input_idx]
    input_to_perturb = _get_input_to_perturb(input)
    wrapped_fn = _with_prepare_inputs(fn, inputs, input_idx, input_to_perturb, True)
    nbhd_checks_fn = functools.partial(_check_outputs_same_dtype_and_shape, eps=eps)
    jvp_fn = _get_numerical_jvp_fn(wrapped_fn, input_to_perturb, eps, nbhd_checks_fn)
    u = _reshape_tensor_or_tuple(u, input_to_perturb.shape)
    u = _mul_tensor_or_tuple(u, eps)
    return _compute_numerical_jvps_wrt_specific_input(jvp_fn, u, input.is_complex(), is_forward_ad)

def _get_numerical_vJu(fn, inputs, inp_indices, func_out, all_u, all_v, eps, is_forward_ad):
    if False:
        while True:
            i = 10
    reduced_jacobians: List[List[torch.Tensor]] = []
    for (i, (inp_idx, u)) in enumerate(zip(inp_indices, all_u)):
        all_Ju = _get_numerical_jvp_wrt_specific_input(fn, inp_idx, inputs, u, eps, is_forward_ad)
        filtered_Ju = []
        func_out = _as_tuple(func_out)
        assert len(all_Ju) == len(func_out)
        for (Ju, output) in zip(all_Ju, func_out):
            if _is_float_or_complex_tensor(output):
                filtered_Ju.append(Ju)
            else:
                pass
        if all_v is not None:
            jacobian_scalars: List[torch.Tensor] = []
            for (v, Ju) in zip(all_v, filtered_Ju):
                jacobian_scalars.append(_dot_with_type_promotion(v, Ju))
            reduced_jacobians.append(jacobian_scalars)
        else:
            reduced_jacobians.append(filtered_Ju)
    return reduced_jacobians

def _check_jacobians_equal(j1, j2, atol):
    if False:
        return 10
    for (j1_x, j2_x) in zip(j1, j2):
        if j1_x.numel() != 0 and (j1_x - j2_x).abs().max() > atol:
            return False
    return True

def _stack_and_check_tensors(list_of_list_of_tensors, inputs, numel_outputs) -> Tuple[Tuple[torch.Tensor, ...], bool, bool]:
    if False:
        for i in range(10):
            print('nop')
    out_jacobians = _allocate_jacobians_with_inputs(inputs, numel_outputs)
    diff_input_list = list(_iter_tensors(inputs, True))
    correct_grad_sizes = True
    correct_grad_types = True
    for (i, tensor_list) in enumerate(list_of_list_of_tensors):
        inp = diff_input_list[i]
        out_jacobian = out_jacobians[i]
        for (j, tensor) in enumerate(tensor_list):
            if tensor is not None and tensor.size() != inp.size():
                correct_grad_sizes = False
            elif tensor is not None and tensor.dtype != inp.dtype:
                correct_grad_types = False
            if tensor is None:
                out_jacobian[:, j].zero_()
            else:
                dense = tensor.to_dense() if not tensor.layout == torch.strided else tensor
                assert out_jacobian[:, j].numel() == dense.numel()
                out_jacobian[:, j] = dense.reshape(-1)
    return (out_jacobians, correct_grad_sizes, correct_grad_types)
FAILED_NONDET_MSG = '\n\nNOTE: If your op relies on non-deterministic operations i.e., it is listed here:\nhttps://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html\nthis failure might be expected.\n\nIf you are adding a new operator, please file an issue and then use one of the\nworkarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.\nIf the test\n- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck\n  with `nondet_tol=<tol>` as a keyword argument.\n- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test\n  to have `gradcheck_nondet_tol=<tol>`.\n- is a Module test (e.g., in common_nn.py), then modify the corresponding\n  module_test entry to have `gradcheck_nondet_tol=<tol>`\n'

def _check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes, fast_mode=False, v=None) -> Tuple[torch.Tensor, ...]:
    if False:
        print('Hello World!')
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        if False:
            return 10
        return torch.autograd.grad(output, diff_input_list, grad_output, retain_graph=True, allow_unused=True)
    if fast_mode:
        vjps1 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
        vjps2 = _get_analytical_vjps_wrt_specific_output(vjp_fn, output.clone(), v)
    else:
        vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
        vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    output_numel = output.numel() if not fast_mode else 1
    (jacobians1, types_ok, sizes_ok) = _stack_and_check_tensors(vjps1, inputs, output_numel)
    (jacobians2, _, _) = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)
    if not types_ok and check_grad_dtypes:
        raise GradcheckError('Gradient has dtype mismatch')
    if not sizes_ok:
        raise GradcheckError('Analytical gradient has incorrect size')
    if not reentrant:
        raise GradcheckError(f'Backward is not reentrant, i.e., running backward with same input and grad_output multiple times gives different values, although analytical gradient matches numerical gradient.The tolerance for nondeterminism was {nondet_tol}.' + FAILED_NONDET_MSG)
    return jacobians1

def _get_analytical_vJu_backward_mode(inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u):
    if False:
        i = 10
        return i + 15
    reduced_jacobians: List[List[torch.Tensor]] = []
    for (output, v) in zip(outputs, all_v):
        all_vJ = _check_analytical_jacobian_attributes(inputs, output, nondet_tol, check_grad_dtypes, fast_mode=True, v=v)
        jacobian_scalars: List[torch.Tensor] = []
        for (vJ, u) in zip(all_vJ, all_u):
            vJ = vJ.T.squeeze(0)
            if vJ.is_complex():
                tv = torch.view_as_real(vJ.resolve_conj())
                tr = tv.select(-1, 0)
                ti = tv.select(-1, 1)
                jacobian_scalars.append(tr.dot(u[0]) + 1j * ti.dot(u[1]))
            else:
                jacobian_scalars.append(vJ.dot(u))
        reduced_jacobians.append(jacobian_scalars)
    return reduced_jacobians

def get_analytical_jacobian(inputs, output, nondet_tol=0.0, grad_out=1.0):
    if False:
        print('Hello World!')
    warnings.warn("get_analytical_jacobian was part of PyTorch's private API and not meant to be exposed. We are deprecating it and it will be removed in a future version of PyTorch. If you have a specific use for this or feature request for this to be a stable API, please file us an issue at https://github.com/pytorch/pytorch/issues/new")
    if grad_out != 1.0:
        raise ValueError('Expected grad_out to be 1.0. get_analytical_jacobian no longer supports values of grad_out != 1.0.')
    if output.is_complex():
        raise ValueError('Expected output to be non-complex. get_analytical_jacobian no longer supports functions that return complex outputs.')
    diff_input_list = list(_iter_tensors(inputs, True))

    def vjp_fn(grad_output):
        if False:
            i = 10
            return i + 15
        return torch.autograd.grad(output, diff_input_list, grad_output, retain_graph=True, allow_unused=True)
    vjps1 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    vjps2 = _compute_analytical_jacobian_rows(vjp_fn, output.clone())
    output_numel = output.numel()
    (jacobians1, types_ok, sizes_ok) = _stack_and_check_tensors(vjps1, inputs, output_numel)
    (jacobians2, _, _) = _stack_and_check_tensors(vjps2, inputs, output_numel)
    reentrant = _check_jacobians_equal(jacobians1, jacobians2, nondet_tol)
    return (jacobians1, reentrant, sizes_ok, types_ok)

def _get_analytical_jacobian(inputs, outputs, input_idx, output_idx):
    if False:
        i = 10
        return i + 15
    jacobians = _check_analytical_jacobian_attributes(inputs, outputs[output_idx], nondet_tol=float('inf'), check_grad_dtypes=False)
    return jacobians[input_idx]

def _compute_analytical_jacobian_rows(vjp_fn, sample_output) -> List[List[Optional[torch.Tensor]]]:
    if False:
        return 10
    grad_out_base = torch.zeros_like(sample_output, memory_format=torch.legacy_contiguous_format)
    flat_grad_out = grad_out_base.view(-1)
    jacobians_rows: List[List[Optional[torch.Tensor]]] = []
    for j in range(flat_grad_out.numel()):
        flat_grad_out.zero_()
        flat_grad_out[j] = 1.0
        grad_inputs = vjp_fn(grad_out_base)
        for (i, d_x) in enumerate(grad_inputs):
            if j == 0:
                jacobians_rows.append([])
            jacobians_rows[i] += [d_x.clone() if isinstance(d_x, torch.Tensor) else None]
    return jacobians_rows

def _get_analytical_vjps_wrt_specific_output(vjp_fn, sample_output, v) -> List[List[Optional[torch.Tensor]]]:
    if False:
        print('Hello World!')
    vjps: List[List[Optional[torch.Tensor]]] = []
    grad_inputs = vjp_fn(v.reshape(sample_output.shape))
    for vjp in grad_inputs:
        vjps.append([vjp.clone() if isinstance(vjp, torch.Tensor) else None])
    return vjps

def _check_inputs(tupled_inputs) -> bool:
    if False:
        while True:
            i = 10
    any_input_requiring_grad = False
    for (idx, inp) in enumerate(tupled_inputs):
        if is_tensor_like(inp) and inp.requires_grad:
            if not (inp.dtype == torch.float64 or inp.dtype == torch.complex128):
                warnings.warn(f'Input #{idx} requires gradient and is not a double precision floating point or complex. This check will likely fail if all the inputs are not of double precision floating point or complex. ')
            if inp.is_sparse:
                content = inp._values()
            elif _is_sparse_compressed_tensor(inp):
                content = inp.values()
            else:
                content = inp
            if content.layout is not torch._mkldnn:
                if not all((st > 0 or sz <= 1 for (st, sz) in zip(content.stride(), content.size()))):
                    raise RuntimeError(f'The {idx}th input has a dimension with stride 0. gradcheck only supports inputs that are non-overlapping to be able to compute the numerical gradients correctly. You should call .contiguous on the input before passing it to gradcheck.')
            any_input_requiring_grad = True
    if not any_input_requiring_grad:
        raise ValueError('gradcheck expects at least one input tensor to require gradient, but none of the them have requires_grad=True.')
    return True

def _check_outputs(outputs) -> None:
    if False:
        return 10
    if any((_is_sparse_any_tensor(t) for t in outputs if isinstance(t, torch.Tensor))):
        raise ValueError('Sparse output is not supported at gradcheck yet. Please call to_dense(masked_grad=...) on the output of fn for gradcheck.')
    if any((t.layout == torch._mkldnn for t in outputs if isinstance(t, torch.Tensor))):
        raise ValueError('MKLDNN output is not supported at gradcheck yet. Please call to_dense(masked_grad=...) on the output of fn for gradcheck.')

def _check_no_differentiable_outputs(func, inputs, func_out, eps, *, is_forward_ad) -> bool:
    if False:
        for i in range(10):
            print('nop')
    jacobians_all_inputs_outputs = _get_numerical_jacobian(func, inputs, func_out, eps=eps, is_forward_ad=is_forward_ad)
    for jacobians_all_outputs_and_fixed_input in jacobians_all_inputs_outputs:
        for jacobian in jacobians_all_outputs_and_fixed_input:
            if torch.ne(jacobian, 0).sum() > 0:
                raise GradcheckError('Numerical gradient for function expected to be zero')
    return True

def _check_no_differentiable_outputs_fast(func, func_out, all_inputs, inputs_indices, all_u, eps, nondet_tol):
    if False:
        i = 10
        return i + 15
    for (inp_idx, u) in zip(inputs_indices, all_u):
        jvps = _get_numerical_jvp_wrt_specific_input(func, inp_idx, all_inputs, u, eps)
        for jvp in jvps:
            if jvp.numel() == 0:
                continue
            if (jvp - torch.zeros_like(jvp)).abs().max() > nondet_tol:
                raise GradcheckError('Numerical gradient for function expected to be zero')
    return True
FAILED_BATCHED_GRAD_MSG = "\ngradcheck or gradgradcheck failed while testing batched gradient computation.\nThis could have been invoked in a number of ways (via a test that calls\ngradcheck/gradgradcheck directly or via an autogenerated test).\n\nIf you are adding a new operator, please file an issue and then use one of the\nworkarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.\nIf the test\n- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck\n  with `check_batched_grad=False` as a keyword argument.\n- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test\n  to have `check_batched_grad=False` and/or `check_batched_gradgrad=False`.\n\nIf you're modifying an existing operator that supports batched grad computation,\nor wish to make a new operator work with batched grad computation, please read\nthe following.\n\nTo compute batched grads (e.g., jacobians, hessians), we vmap over the backward\ncomputation. The most common failure case is if there is a 'vmap-incompatible\noperation' in the backward pass. Please see\nNOTE: [How to write vmap-compatible backward formulas]\nin the codebase for an explanation of how to fix this.\n".strip()
FAILED_BATCHED_GRAD_MSG_FWD_AD = '\ngradcheck failed while testing batched gradient computation with forward-mode AD.\nThis test is enabled automatically when both `check_batched_grad=True`\nand `check_forward_ad=True`, but can be disabled in the following ways\ndependong on how the test was invoked (via a test that calls gradcheck\ndirectly or via an autogenerated test).\n\nIf you are adding a new operator, please file an issue and then use one of the\nworkarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck.\nIf the test\n- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck\n  with `check_batched_forward_grad=False` as a keyword argument.\n- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test\n  to have `check_batched_forward_grad=False`\n'

def _get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp, is_forward_ad=False):
    if False:
        while True:
            i = 10
    return f'\nFor output {output_idx} and input {input_idx}:\n\n{(FAILED_BATCHED_GRAD_MSG_FWD_AD if is_forward_ad else FAILED_BATCHED_GRAD_MSG)}\n\nGot:\n{res}\n\nExpected:\n{exp}\n'.strip()

def _test_batched_grad_forward_ad(func, inputs) -> bool:
    if False:
        print('Hello World!')
    fwAD = torch.autograd.forward_ad
    assert isinstance(inputs, tuple)
    for (input_idx, current_input) in enumerate(inputs):
        if not (is_tensor_like(current_input) and current_input.requires_grad):
            continue

        def jvp(tangent: torch.Tensor):
            if False:
                print('Hello World!')
            with fwAD.dual_level():
                dual = fwAD.make_dual(current_input.detach(), tangent)
                inputs_with_dual = tuple((dual if idx == input_idx else inp.detach() if is_tensor_like(inp) else inp for (idx, inp) in enumerate(inputs)))
                dual_outputs = _as_tuple(func(*inputs_with_dual))
                ret = []
                for dual_output in dual_outputs:
                    if dual_output is None:
                        continue
                    (primal_out, tangent_out) = fwAD.unpack_dual(dual_output)
                    if tangent_out is not None:
                        ret.append(tangent_out)
                    else:
                        ret.append(torch.zeros([], dtype=primal_out.dtype, device=primal_out.device).expand(primal_out.shape))
                return tuple(ret)
        if not _is_float_or_complex_tensor(current_input):
            continue
        tangents = [torch.randn_like(current_input) for _ in range(2)]
        expected = [jvp(t) for t in tangents]
        expected = [torch.stack(shards) for shards in zip(*expected)]
        try:
            result = _vmap(jvp)(torch.stack(tangents))
        except RuntimeError as ex:
            raise GradcheckError(f'While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG_FWD_AD}') from ex
        for (input_idx, (res, exp)) in enumerate(zip(result, expected)):
            if torch.allclose(res, exp):
                continue
            raise GradcheckError(_get_failed_batched_grad_test_msg(input_idx, input_idx, res, exp, is_forward_ad=True))
    return True

def _test_batched_grad(input, output, output_idx) -> bool:
    if False:
        print('Hello World!')
    diff_input_list = list(_iter_tensors(input, True))
    grad = functools.partial(torch.autograd.grad, output, diff_input_list, retain_graph=True, allow_unused=True)

    def vjp(v):
        if False:
            i = 10
            return i + 15
        results = grad(v)
        results = tuple((grad if grad is not None else torch.zeros([], dtype=inp.dtype, device=inp.device).expand(inp.shape) for (grad, inp) in zip(results, diff_input_list)))
        return results
    grad_outputs = [torch.randn_like(output) for _ in range(2)]
    expected = [vjp(gO) for gO in grad_outputs]
    expected = [torch.stack(shards) for shards in zip(*expected)]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='There is a performance drop')
        warnings.filterwarnings('ignore', message='Please use torch.vmap')
        try:
            result = vmap(vjp)(torch.stack(grad_outputs))
        except RuntimeError as ex:
            raise GradcheckError(f'While computing batched gradients, got: {ex}\n\n{FAILED_BATCHED_GRAD_MSG}') from ex
    for (input_idx, (res, exp)) in enumerate(zip(result, expected)):
        if torch.allclose(res, exp):
            continue
        raise GradcheckError(_get_failed_batched_grad_test_msg(output_idx, input_idx, res, exp))
    return True

def _test_backward_mul_by_grad_output(outputs, inputs, masked) -> bool:
    if False:
        print('Hello World!')
    diff_input_list: List[torch.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError('no Tensors requiring grad found in input')
    grads_input = torch.autograd.grad(outputs, diff_input_list, [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in outputs], allow_unused=True)
    for (gi, di) in zip(grads_input, diff_input_list):
        if gi is None:
            continue
        if isinstance(gi, torch.Tensor) and gi.layout != torch.strided:
            if gi.layout != di.layout:
                raise GradcheckError('grad is incorrect layout (' + str(gi.layout) + ' is not ' + str(di.layout) + ')')
            if _is_sparse_any_tensor(gi):
                sparse_kind = str(gi.layout).replace('torch.', '').replace('_coo', '')
                if gi.sparse_dim() != di.sparse_dim():
                    raise GradcheckError(f'grad is {sparse_kind} tensor, but has incorrect sparse_dim {gi.sparse_dim()}, expected {di.sparse_dim()}')
                if gi.dense_dim() != di.dense_dim():
                    raise GradcheckError(f'grad is {sparse_kind} tensor, but has incorrect dense_dim {gi.dense_dim()}, expected {di.dense_dim()}')
            gi = gi.to_dense()
            di = di.to_dense()
        if masked:
            if not torch.allclose(gi, torch.zeros_like(gi)):
                raise GradcheckError('backward not multiplied by grad_output')
        elif not gi.eq(0).all():
            raise GradcheckError('backward not multiplied by grad_output')
        if gi.dtype != di.dtype:
            raise GradcheckError('grad is incorrect type')
        if gi.device != di.device:
            raise GradcheckError('grad is incorrect device')
        if gi.size() != di.size():
            raise GradcheckError('grad is incorrect size')
    return True

def _test_undefined_forward_mode(func, outputs, inputs):
    if False:
        return 10
    fwAD = torch.autograd.forward_ad
    (inp_tensors_idx, inp_tensors) = _get_inp_tensors(inputs)
    (all_v, all_u, all_u_dense) = _make_vectors(inp_tensors, outputs, use_forward_ad=True)
    tensor_inputs = tuple((i for i in inputs if is_tensor_like(i) and i.requires_grad))
    with fwAD.dual_level():
        fw_grads = []
        dual_inputs = []
        tensor_indices = set()
        for (i, inp) in enumerate(inputs):
            if is_tensor_like(inp) and inp.requires_grad:
                if inp.layout == torch._mkldnn:
                    raise ValueError('MKLDNN inputs are not support for forward AD gradcheck.')
                inp = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
                fw_grads.append(fwAD.unpack_dual(inp)[1])
                tensor_indices.add(i)
            dual_inputs.append(inp)
        for (i, (fw_grad, u)) in enumerate(zip(fw_grads, all_u)):
            fw_grad.copy_(u.view_as(fw_grad))
        for (idx, inp) in enumerate(inputs):
            if idx not in tensor_indices:
                continue
            dual_inp_obj = dual_inputs[idx]
            dual_inputs[idx] = fwAD.make_dual(inp.detach(), torch.zeros_like(inp))
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs1 = filter(_is_float_or_complex_tensor, raw_outputs)
            dual_inputs[idx] = inp.detach()
            raw_outputs = _as_tuple(func(*dual_inputs))
            dual_outputs2 = filter(_is_float_or_complex_tensor, raw_outputs)
            dual_inputs[idx] = dual_inp_obj
            for (index_o, (d_o1, d_o2)) in enumerate(zip(dual_outputs1, dual_outputs2)):
                (val1, res1) = fwAD.unpack_dual(d_o1)
                (val2, res2) = fwAD.unpack_dual(d_o2)
                if not (res1 is None or res2 is None):
                    if not torch.allclose(res1, res2):
                        raise GradcheckError('Mismatch in tangent values for output with index: ', index_o, ' when input: ', inp, ' has an undefined tangent value. ', ' Got: ', res1, ' but expected: ', res2)
    return True

def _test_undefined_backward_mode(func, outputs, inputs) -> bool:
    if False:
        for i in range(10):
            print('nop')
    diff_input_list: List[torch.Tensor] = list(_iter_tensors(inputs, True))
    if not diff_input_list:
        raise GradcheckError('no Tensors requiring grad found in input')

    def warn_bc_breaking():
        if False:
            for i in range(10):
                print('nop')
        warnings.warn('Backwards compatibility: New undefined gradient support checking feature is enabled by default, but it may break existing callers of this function. If this is true for you, you can call this function with "check_undefined_grad=False" to disable the feature')

    def check_undefined_grad_support(output_to_check):
        if False:
            i = 10
            return i + 15
        grads_output = [torch.zeros_like(o, memory_format=torch.legacy_contiguous_format) for o in output_to_check]
        try:
            grads_input = torch.autograd.grad(output_to_check, diff_input_list, grads_output, allow_unused=True)
        except RuntimeError as e:
            warn_bc_breaking()
            raise GradcheckError('Expected backward function to handle undefined output grads. Please look at "Notes about undefined output gradients" in "tools/autograd/derivatives.yaml"') from e
        for (gi, i) in zip(grads_input, diff_input_list):
            if gi is not None and (not gi.eq(0).all()):
                warn_bc_breaking()
                raise GradcheckError('Expected all input grads to be undefined or zero when all output grads are undefined or zero. Please look at "Notes about undefined output gradients" in "tools/autograd/derivatives.yaml"')
        return True
    outputs_to_check = [[torch._C._functions.UndefinedGrad()(o) for o in _differentiable_outputs(func(*inputs)) if isinstance(o, torch.Tensor)]]
    if len(outputs_to_check[0]) > 1:
        for undef_grad_idx in range(len(outputs)):
            output_to_check = _differentiable_outputs(func(*inputs))
            outputs_to_check.append([torch._C._functions.UndefinedGrad()(o) if idx == undef_grad_idx else o for (idx, o) in enumerate(output_to_check)])
    return all((check_undefined_grad_support(output) for output in outputs_to_check))

def _as_tuple(x):
    if False:
        print('Hello World!')
    if isinstance(x, tuple):
        return x
    elif isinstance(x, list):
        return tuple(x)
    else:
        return (x,)

def _differentiable_outputs(x):
    if False:
        print('Hello World!')
    return tuple((o for o in _as_tuple(x) if o.requires_grad))

def _get_notallclose_msg(analytical, numerical, output_idx, input_idx, complex_indices, test_imag=False, is_forward_ad=False) -> str:
    if False:
        while True:
            i = 10
    out_is_complex = not is_forward_ad and complex_indices and (output_idx in complex_indices)
    inp_is_complex = is_forward_ad and complex_indices and (input_idx in complex_indices)
    part = 'imaginary' if test_imag else 'real'
    element = 'inputs' if is_forward_ad else 'outputs'
    prefix = '' if not (out_is_complex or inp_is_complex) else f'While considering the {part} part of complex {element} only, '
    mode = 'computed with forward mode ' if is_forward_ad else ''
    return prefix + 'Jacobian %smismatch for output %d with respect to input %d,\nnumerical:%s\nanalytical:%s\n' % (mode, output_idx, input_idx, numerical, analytical)

def _transpose(matrix_of_tensors):
    if False:
        print('Hello World!')
    return list(zip(*matrix_of_tensors))

def _real_and_imag_output(fn):
    if False:
        for i in range(10):
            print('nop')

    def apply_to_c_outs(fn, fn_to_apply):
        if False:
            while True:
                i = 10

        def wrapped_fn(*inputs):
            if False:
                while True:
                    i = 10
            outs = _as_tuple(fn(*inputs))
            return tuple((fn_to_apply(o) if o.is_complex() else o for o in outs))
        return wrapped_fn
    return (apply_to_c_outs(fn, torch.real), apply_to_c_outs(fn, torch.imag))

def _real_and_imag_input(fn, complex_inp_indices, tupled_inputs):
    if False:
        return 10

    def apply_to_c_inps(fn, fn_to_apply):
        if False:
            print('Hello World!')

        def wrapped_fn(*inputs):
            if False:
                while True:
                    i = 10
            new_inputs = list(inputs)
            for should_be_complex in complex_inp_indices:
                new_inputs[should_be_complex] = fn_to_apply(new_inputs[should_be_complex], tupled_inputs[should_be_complex])
            return _as_tuple(fn(*new_inputs))
        return wrapped_fn
    real_fn = apply_to_c_inps(fn, lambda inp, orig: inp + orig.imag * 1j)
    imag_fn = apply_to_c_inps(fn, lambda inp, orig: orig.real + inp * 1j)
    return (real_fn, imag_fn)

def _gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, check_forward_ad, check_backward_ad, nondet_tol, check_undefined_grad):
    if False:
        print('Hello World!')
    complex_out_indices = [i for (i, o) in enumerate(outputs) if o.is_complex()]
    has_any_complex_output = any((o.is_complex() for o in _as_tuple(func_out)))
    if check_backward_ad:
        if has_any_complex_output:
            (real_fn, imag_fn) = _real_and_imag_output(func)
            imag_func_out = imag_fn(*tupled_inputs)
            imag_outputs = _differentiable_outputs(imag_func_out)
            gradcheck_fn(imag_fn, imag_func_out, tupled_inputs, imag_outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_out_indices, test_imag=True)
            real_func_out = real_fn(*tupled_inputs)
            real_outputs = _differentiable_outputs(real_func_out)
            gradcheck_fn(real_fn, real_func_out, tupled_inputs, real_outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_out_indices)
        else:
            gradcheck_fn(func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol)
    if check_forward_ad:
        complex_inp_indices = [i for (i, inp) in enumerate(tupled_inputs) if is_tensor_like(inp) and inp.is_complex()]
        if complex_inp_indices:
            (real_fn, imag_fn) = _real_and_imag_input(func, complex_inp_indices, tupled_inputs)
            imag_inputs = [inp.imag if is_tensor_like(inp) and inp.is_complex() else inp for inp in tupled_inputs]
            imag_func_out = imag_fn(*imag_inputs)
            diff_imag_func_out = _differentiable_outputs(imag_func_out)
            gradcheck_fn(imag_fn, imag_func_out, imag_inputs, diff_imag_func_out, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_inp_indices, test_imag=True, use_forward_ad=True)
            real_inputs = [inp.real if is_tensor_like(inp) and inp.is_complex() else inp for inp in tupled_inputs]
            real_func_out = real_fn(*real_inputs)
            diff_real_func_out = _differentiable_outputs(real_func_out)
            gradcheck_fn(real_fn, real_func_out, real_inputs, diff_real_func_out, eps, rtol, atol, check_grad_dtypes, nondet_tol, complex_indices=complex_inp_indices, use_forward_ad=True)
            if check_undefined_grad:
                _test_undefined_forward_mode(imag_fn, imag_func_out, imag_inputs)
                _test_undefined_forward_mode(real_fn, real_func_out, real_inputs)
        else:
            gradcheck_fn(func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, use_forward_ad=True)
            if check_undefined_grad:
                _test_undefined_forward_mode(func, outputs, tupled_inputs)

def _slow_gradcheck(func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, *, use_forward_ad=False, complex_indices=None, test_imag=False, masked=False):
    if False:
        for i in range(10):
            print('nop')
    func_out = _as_tuple(func_out)
    if not outputs:
        return _check_no_differentiable_outputs(func, tupled_inputs, func_out, eps=eps, is_forward_ad=use_forward_ad)
    tupled_inputs_numerical = tupled_inputs if masked else _densify(tupled_inputs)
    numerical = _transpose(_get_numerical_jacobian(func, tupled_inputs_numerical, func_out, eps=eps, is_forward_ad=use_forward_ad))
    numerical = [nj for (o, nj) in zip(func_out, numerical) if o.requires_grad]
    if use_forward_ad:
        analytical_forward = _get_analytical_jacobian_forward_ad(func, tupled_inputs, func_out, check_grad_dtypes=check_grad_dtypes)
        for (i, n_per_out) in enumerate(numerical):
            for (j, n) in enumerate(n_per_out):
                a = analytical_forward[j][i]
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(_get_notallclose_msg(a, n, i, j, complex_indices, test_imag, is_forward_ad=True))
    else:
        for (i, o) in enumerate(outputs):
            analytical = _check_analytical_jacobian_attributes(tupled_inputs, o, nondet_tol, check_grad_dtypes)
            for (j, (a, n)) in enumerate(zip(analytical, numerical[i])):
                if not _allclose_with_type_promotion(a, n.to(a.device), rtol, atol):
                    raise GradcheckError(_get_notallclose_msg(a, n, i, j, complex_indices, test_imag))
    return True

def _dot_with_type_promotion(u, v):
    if False:
        i = 10
        return i + 15
    assert u.dim() == 1 and v.dim() == 1
    return (u * v).sum()

def _allclose_with_type_promotion(a, b, rtol, atol):
    if False:
        return 10
    promoted_type = torch.promote_types(a.dtype, b.dtype)
    a = a.to(dtype=promoted_type)
    b = b.to(dtype=promoted_type)
    return torch.allclose(a, b, rtol, atol)

def _to_real_dtype(dtype):
    if False:
        while True:
            i = 10
    if dtype == torch.complex128:
        return torch.float64
    elif dtype == torch.complex64:
        return torch.float32
    else:
        return dtype

def _vec_from_tensor(x, generator, downcast_complex=False):
    if False:
        print('Hello World!')
    if x.layout == torch.sparse_coo:
        x_values = x._values()
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        values = torch.rand(x_values.numel(), generator=generator).to(dtype=dtype, device=x.device).view(x_values.shape)
        values /= values.norm()
        vec = torch.sparse_coo_tensor(x._indices(), values, x.size())
    elif _is_sparse_compressed_tensor(x):
        if x.layout in {torch.sparse_csr, torch.sparse_bsr}:
            (compressed_indices, plain_indices) = (x.crow_indices(), x.col_indices())
        else:
            (compressed_indices, plain_indices) = (x.ccol_indices(), x.row_indices())
        x_values = x.values()
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        values = torch.rand(x_values.numel(), generator=generator).to(dtype=dtype, device=x.device).view(x_values.shape)
        values /= values.norm()
        vec = torch.sparse_compressed_tensor(compressed_indices, plain_indices, values, x.size(), layout=x.layout)
    else:
        dtype = _to_real_dtype(x.dtype) if downcast_complex else x.dtype
        vec = torch.rand(x.numel(), generator=generator).to(dtype=dtype, device=x.device)
        vec /= vec.norm()
    return vec

def _get_inp_tensors(tupled_inputs):
    if False:
        print('Hello World!')
    inp_idx_tup = [(i, t) for (i, t) in enumerate(tupled_inputs) if is_tensor_like(t) and t.requires_grad]
    return ([tup[0] for tup in inp_idx_tup], [tup[1] for tup in inp_idx_tup])

def _adjusted_atol(atol, u, v):
    if False:
        for i in range(10):
            print('nop')
    u = u[0] if isinstance(u, tuple) else u
    sum_u = u.sum()
    sum_v = 1.0 if v is None else v.sum()
    return atol * float(sum_u) * float(sum_v)
FAST_FAIL_SLOW_OK_MSG = "\nFast gradcheck failed but element-wise differences are small. This means that the\ntest might've passed in slow_mode!\n\nIf you are adding a new operator, please file an issue and then use one of the\nworkarounds. The workaround depends on how your test invokes gradcheck/gradgradcheck:\n\nIf the test\n- manually invokes gradcheck/gradgradcheck, then call gradcheck/gradgradcheck\n  with `fast_mode=False` as a keyword argument.\n- is OpInfo-based (e.g., in test_ops_gradients.py), then modify the OpInfo for the test\n  to have `gradcheck_fast_mode=False`\n- is a Module test (e.g., in common_nn.py), then modify the corresponding\n  module_test entry to have `gradcheck_fast_mode=False`\n".strip()

def _run_slow_mode_and_get_error(func, tupled_inputs, outputs, input_idx, output_idx, rtol, atol, is_forward_ad):
    if False:
        while True:
            i = 10
    slow_numerical = _get_numerical_jacobian(func, tupled_inputs, outputs, is_forward_ad=is_forward_ad)[input_idx][output_idx]
    if is_forward_ad:

        def new_fn(inp):
            if False:
                while True:
                    i = 10
            new_inputs = list(tupled_inputs)
            new_inputs[input_idx] = inp
            return _as_tuple(func(*new_inputs))[output_idx]
        slow_analytical = _get_analytical_jacobian_forward_ad(new_fn, (tupled_inputs[input_idx],), (outputs[output_idx],))[0][0]
    else:
        slow_analytical = _get_analytical_jacobian(tupled_inputs, outputs, input_idx, output_idx)
    slow_max_diff = (slow_numerical - slow_analytical).abs().max()
    slow_allclose = torch.allclose(slow_analytical, slow_numerical, rtol, atol)
    msg = f'\nThe above quantities relating the numerical and analytical jacobians are computed \nin fast mode. See: https://github.com/pytorch/pytorch/issues/53876 for more background \nabout fast mode. Below, we recompute numerical and analytical jacobians in slow mode:\n\nNumerical:\n {slow_numerical}\nAnalytical:\n{slow_analytical}\n\nThe max per-element difference (slow mode) is: {slow_max_diff}.\n'
    if slow_allclose:
        msg += FAST_FAIL_SLOW_OK_MSG
    return msg

def _to_flat_dense_if_sparse(tensor):
    if False:
        return 10
    if _is_sparse_any_tensor(tensor):
        return tensor.to_dense().reshape(-1)
    else:
        return tensor

def _make_vectors(inp_tensors, outputs, *, use_forward_ad):
    if False:
        i = 10
        return i + 15
    g_cpu = torch.Generator()
    all_u = []
    all_u_dense = []
    for inp in inp_tensors:
        ur = _vec_from_tensor(inp, g_cpu, True)
        ur_dense = _to_flat_dense_if_sparse(ur)
        if inp.is_complex():
            ui = _vec_from_tensor(inp, g_cpu, True)
            all_u.append((ur, ui))
            ui_dense = _to_flat_dense_if_sparse(ui)
            all_u_dense.append((ur_dense, ui_dense))
        else:
            all_u.append(ur)
            all_u_dense.append(ur_dense)
    all_v = None if use_forward_ad else [_vec_from_tensor(out, g_cpu) for out in outputs]
    return (all_v, all_u, all_u_dense)

def _check_analytical_numerical_equal(all_analytical, all_numerical, complex_indices, tupled_inputs, outputs, func, all_v, all_u, rtol, atol, test_imag, *, is_forward_ad=False):
    if False:
        print('Hello World!')
    for (i, all_numerical_for_input_i) in enumerate(all_numerical):
        for (j, n) in enumerate(all_numerical_for_input_i):
            if is_forward_ad:
                a = all_analytical[i][j]
            else:
                a = all_analytical[j][i]
            n = n.to(device=a.device)
            updated_atol = _adjusted_atol(atol, all_u[i], all_v[j] if all_v else None)
            if not _allclose_with_type_promotion(a, n.to(a.device), rtol, updated_atol):
                jacobians_str = _run_slow_mode_and_get_error(func, tupled_inputs, outputs, i, j, rtol, atol, is_forward_ad)
                raise GradcheckError(_get_notallclose_msg(a, n, j, i, complex_indices, test_imag, is_forward_ad) + jacobians_str)

def _fast_gradcheck(func, func_out, inputs, outputs, eps, rtol, atol, check_grad_dtypes, nondet_tol, *, use_forward_ad=False, complex_indices=None, test_imag=False, masked=False):
    if False:
        print('Hello World!')
    (inp_tensors_idx, inp_tensors) = _get_inp_tensors(inputs)
    (all_v, all_u, all_u_dense) = _make_vectors(inp_tensors, outputs, use_forward_ad=use_forward_ad)
    (inputs_numerical, all_u_numerical, all_v_numerical) = (inputs, all_u, all_v) if masked else _densify((inputs, all_u, all_v))
    numerical_vJu = _get_numerical_vJu(func, inputs_numerical, inp_tensors_idx, func_out, all_u_numerical, all_v_numerical, eps, is_forward_ad=use_forward_ad)
    if use_forward_ad:
        assert all_v is None
        analytical_vJu = _get_analytical_jacobian_forward_ad(func, inputs, _as_tuple(func_out), all_u=all_u, check_grad_dtypes=check_grad_dtypes)
    else:
        if not outputs:
            _check_no_differentiable_outputs_fast(func, func_out, inputs, inp_tensors_idx, all_u, eps, nondet_tol)
        analytical_vJu = _get_analytical_vJu_backward_mode(inputs, outputs, nondet_tol, check_grad_dtypes, all_v, all_u_dense)
    _check_analytical_numerical_equal(analytical_vJu, numerical_vJu, complex_indices, inputs, outputs, func, all_v, all_u, rtol, atol, test_imag, is_forward_ad=use_forward_ad)
    return True

def gradcheck(func: Callable[..., Union[_TensorOrTensors]], inputs: _TensorOrTensors, *, eps: float=1e-06, atol: float=1e-05, rtol: float=0.001, raise_exception: bool=True, check_sparse_nnz: Optional[bool]=None, nondet_tol: float=0.0, check_undefined_grad: bool=True, check_grad_dtypes: bool=False, check_batched_grad: bool=False, check_batched_forward_grad: bool=False, check_forward_ad: bool=False, check_backward_ad: bool=True, fast_mode: bool=False, masked: Optional[bool]=None) -> bool:
    if False:
        while True:
            i = 10
    'Check gradients computed via small finite differences against analytical\n    gradients wrt tensors in :attr:`inputs` that are of floating point or complex type\n    and with ``requires_grad=True``.\n\n    The check between numerical and analytical gradients uses :func:`~torch.allclose`.\n\n    For most of the complex functions we consider for optimization purposes, no notion of\n    Jacobian exists. Instead, gradcheck verifies if the numerical and analytical values of\n    the Wirtinger and Conjugate Wirtinger derivatives are consistent. Because the gradient\n    computation is done under the assumption that the overall function has a real-valued\n    output, we treat functions with complex output in a special way. For these functions,\n    gradcheck is applied to two real-valued functions corresponding to taking the real\n    components of the complex outputs for the first, and taking the imaginary components\n    of the complex outputs for the second. For more details, check out\n    :ref:`complex_autograd-doc`.\n\n    .. note::\n        The default values are designed for :attr:`input` of double precision.\n        This check will likely fail if :attr:`input` is of less precision, e.g.,\n        ``FloatTensor``.\n\n    .. note::\n        Gradcheck may fail when evaluated on non-differentiable points\n        because the numerically computed gradients via finite differencing may differ\n        those computed analytically (not necessarily because either is incorrect).\n        For more context, see :ref:`non-differentiable-func-grad`.\n\n    .. warning::\n       If any checked tensor in :attr:`input` has overlapping memory, i.e.,\n       different indices pointing to the same memory address (e.g., from\n       :func:`torch.expand`), this check will likely fail because the numerical\n       gradients computed by point perturbation at such indices will change\n       values at all other indices that share the same memory address.\n\n    Args:\n        func (function): a Python function that takes Tensor inputs and returns\n            a Tensor or a tuple of Tensors\n        inputs (tuple of Tensor or Tensor): inputs to the function\n        eps (float, optional): perturbation for finite differences\n        atol (float, optional): absolute tolerance\n        rtol (float, optional): relative tolerance\n        raise_exception (bool, optional): indicating whether to raise an exception if\n            the check fails. The exception gives more information about the\n            exact nature of the failure. This is helpful when debugging gradchecks.\n        check_sparse_nnz (bool, optional): if ``True``, gradcheck allows\n            for SparseTensor input, and for any SparseTensor inputs,\n            gradcheck will perform its check at ``nnz`` positions only.\n            The ``check_sparse_nnz`` argument is deprecated, use the\n            ``masked`` argument instead. If ``check_sparse_nnz != masked``, an\n            exception is raised.\n        nondet_tol (float, optional): tolerance for non-determinism. When running\n            identical inputs through the differentiation, the results must either match\n            exactly (default, 0.0) or be within this tolerance.\n        check_undefined_grad (bool, optional): if ``True``, check if undefined output grads\n            are supported and treated as zeros, for ``Tensor`` outputs.\n        check_batched_grad (bool, optional): if ``True``, check if we can compute\n            batched gradients using prototype vmap support. Defaults to False.\n        check_batched_forward_grad (bool, optional): if ``True``, checks if we can compute\n            batched forward gradients using forward ad and prototype vmap support. Defaults to ``False``.\n        check_forward_ad (bool, optional): if ``True``, check that the gradients computed with forward\n            mode AD match the numerical ones. Defaults to ``False``.\n        check_backward_ad (bool, optional): if ``False``, do not perform any checks that rely on\n            backward mode AD to be implemented. Defaults to ``True``.\n        fast_mode (bool, optional): Fast mode for gradcheck and gradgradcheck is currently only\n            implemented for R to R functions. If none of the inputs and outputs are complex\n            a faster implementation of gradcheck that no longer computes the entire jacobian\n            is run; otherwise, we fall back to the slow implementation.\n        masked (bool, optional): if ``True``, the gradients of unspecified elements of\n            sparse tensors are ignored. Defaults to ``False``.\n    Returns:\n        ``True`` if all differences satisfy allclose condition\n\n    '
    if check_sparse_nnz is None:
        if masked is None:
            check_sparse_nnz = masked = False
        else:
            check_sparse_nnz = masked
    else:
        warnings.warn(f'Backwards compatibility: check_sparse_nnz is deprecated, it will be removed in a future version of PyTorch. Use masked={check_sparse_nnz} instead.')
        if masked is None:
            masked = check_sparse_nnz
        elif check_sparse_nnz != masked:
            raise ValueError(f'Expected specified check_sparse_nnz (={check_sparse_nnz}) to be equal to masked (={masked}).')
    assert check_forward_ad or check_backward_ad, 'Expected at least one of check_forward_ad or check_backward_ad to be True'
    assert not (check_batched_grad and (not check_backward_ad)), 'Setting check_batched_grad=True requires check_backward_ad to be True'
    assert not (check_batched_forward_grad and (not check_forward_ad)), 'Setting check_batched_forward_grad=True requires check_forward_ad to be True'
    args = locals().copy()
    args.pop('raise_exception')
    args.pop('check_sparse_nnz')
    if not raise_exception:
        try:
            return _gradcheck_helper(**args)
        except GradcheckError as e:
            return False
    else:
        return _gradcheck_helper(**args)

def _gradcheck_helper(func, inputs, eps, atol, rtol, nondet_tol, check_undefined_grad, check_grad_dtypes, check_batched_grad, check_batched_forward_grad, check_forward_ad, check_backward_ad, fast_mode, masked):
    if False:
        print('Hello World!')
    tupled_inputs = _as_tuple(inputs)
    _check_inputs(tupled_inputs)
    func_out = func(*tupled_inputs)
    outputs = _differentiable_outputs(func_out)
    _check_outputs(outputs)
    gradcheck_fn = functools.partial(_fast_gradcheck if fast_mode else _slow_gradcheck, masked=masked)
    _gradcheck_real_imag(gradcheck_fn, func, func_out, tupled_inputs, outputs, eps, rtol, atol, check_grad_dtypes, check_forward_ad=check_forward_ad, check_backward_ad=check_backward_ad, nondet_tol=nondet_tol, check_undefined_grad=check_undefined_grad)
    if check_batched_forward_grad:
        _test_batched_grad_forward_ad(func, tupled_inputs)
    if not check_backward_ad:
        return True
    for (i, o) in enumerate(outputs):
        if check_batched_grad:
            _test_batched_grad(tupled_inputs, o, i)
    _test_backward_mul_by_grad_output(outputs, tupled_inputs, masked)
    if check_undefined_grad and check_backward_ad:
        _test_undefined_backward_mode(func, outputs, tupled_inputs)
    return True

def gradgradcheck(func: Callable[..., _TensorOrTensors], inputs: _TensorOrTensors, grad_outputs: Optional[_TensorOrTensors]=None, *, eps: float=1e-06, atol: float=1e-05, rtol: float=0.001, gen_non_contig_grad_outputs: bool=False, raise_exception: bool=True, nondet_tol: float=0.0, check_undefined_grad: bool=True, check_grad_dtypes: bool=False, check_batched_grad: bool=False, check_fwd_over_rev: bool=False, check_rev_over_rev: bool=True, fast_mode: bool=False, masked: bool=False) -> bool:
    if False:
        print('Hello World!')
    "Check gradients of gradients computed via small finite differences\n    against analytical gradients wrt tensors in :attr:`inputs` and\n    :attr:`grad_outputs` that are of floating point or complex type and with\n    ``requires_grad=True``.\n\n    This function checks that backpropagating through the gradients computed\n    to the given :attr:`grad_outputs` are correct.\n\n    The check between numerical and analytical gradients uses :func:`~torch.allclose`.\n\n    .. note::\n        The default values are designed for :attr:`input` and\n        :attr:`grad_outputs` of double precision. This check will likely fail if\n        they are of less precision, e.g., ``FloatTensor``.\n\n    .. warning::\n       If any checked tensor in :attr:`input` and :attr:`grad_outputs` has\n       overlapping memory, i.e., different indices pointing to the same memory\n       address (e.g., from :func:`torch.expand`), this check will likely fail\n       because the numerical gradients computed by point perturbation at such\n       indices will change values at all other indices that share the same\n       memory address.\n\n    Args:\n        func (function): a Python function that takes Tensor inputs and returns\n            a Tensor or a tuple of Tensors\n        inputs (tuple of Tensor or Tensor): inputs to the function\n        grad_outputs (tuple of Tensor or Tensor, optional): The gradients with\n            respect to the function's outputs.\n        eps (float, optional): perturbation for finite differences\n        atol (float, optional): absolute tolerance\n        rtol (float, optional): relative tolerance\n        gen_non_contig_grad_outputs (bool, optional): if :attr:`grad_outputs` is\n            ``None`` and :attr:`gen_non_contig_grad_outputs` is ``True``, the\n            randomly generated gradient outputs are made to be noncontiguous\n        raise_exception (bool, optional): indicating whether to raise an exception if\n            the check fails. The exception gives more information about the\n            exact nature of the failure. This is helpful when debugging gradchecks.\n        nondet_tol (float, optional): tolerance for non-determinism. When running\n            identical inputs through the differentiation, the results must either match\n            exactly (default, 0.0) or be within this tolerance. Note that a small amount\n            of nondeterminism in the gradient will lead to larger inaccuracies in\n            the second derivative.\n        check_undefined_grad (bool, optional): if True, check if undefined output grads\n            are supported and treated as zeros\n        check_batched_grad (bool, optional): if True, check if we can compute\n            batched gradients using prototype vmap support. Defaults to False.\n        fast_mode (bool, optional): if True, run a faster implementation of gradgradcheck that\n            no longer computes the entire jacobian.\n        masked (bool, optional): if True, the gradients of unspecified elements of\n            sparse tensors are ignored (default, False).\n    Returns:\n        True if all differences satisfy allclose condition\n    "
    assert check_fwd_over_rev or check_rev_over_rev, 'Expected at least one of check_fwd_over_rev or check_rev_over_rev to be True'
    assert not (check_undefined_grad and (not check_rev_over_rev)), 'Setting check_undefined_grad=True requires check_rev_over_rev to be True'
    assert not (check_batched_grad and (not check_rev_over_rev)), 'Setting check_batched_grad=True requires check_rev_over_rev to be True'
    tupled_inputs = _as_tuple(inputs)
    if grad_outputs is None:
        outputs = _differentiable_outputs(func(*tupled_inputs))
        tupled_grad_outputs = tuple((torch.testing.make_tensor(x.shape, dtype=x.dtype if x.is_floating_point() or x.is_complex() else torch.double, device=x.device, low=-1, high=1, requires_grad=True, noncontiguous=gen_non_contig_grad_outputs) for x in outputs))
    else:
        tupled_grad_outputs = _as_tuple(grad_outputs)
    num_outputs = len(tupled_grad_outputs)
    diff_input_args_indices = {i for (i, x) in enumerate(tupled_inputs) if is_tensor_like(x) and x.requires_grad}
    diff_grad_output_indices = {i for (i, x) in enumerate(tupled_grad_outputs) if x.requires_grad}

    def new_func(*args):
        if False:
            for i in range(10):
                print('nop')
        input_args = tuple((x.requires_grad_() if i in diff_input_args_indices else x for (i, x) in enumerate(args[:-num_outputs])))
        outputs = _differentiable_outputs(func(*input_args))
        grad_outputs = tuple((x.requires_grad_() if i in diff_grad_output_indices else x for (i, x) in enumerate(args[-num_outputs:])))
        diff_input_args = tuple((x for (i, x) in enumerate(input_args) if i in diff_input_args_indices))
        grad_inputs = torch.autograd.grad(outputs, diff_input_args, grad_outputs, create_graph=True, allow_unused=True)
        grad_inputs = tuple((g for g in grad_inputs if g is not None))
        return grad_inputs
    return gradcheck(new_func, tupled_inputs + tupled_grad_outputs, eps=eps, atol=atol, rtol=rtol, raise_exception=raise_exception, nondet_tol=nondet_tol, check_undefined_grad=check_undefined_grad, check_grad_dtypes=check_grad_dtypes, check_batched_grad=check_batched_grad, fast_mode=fast_mode, check_forward_ad=check_fwd_over_rev, check_backward_ad=check_rev_over_rev, masked=masked)