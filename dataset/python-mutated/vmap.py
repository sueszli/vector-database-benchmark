import torch
import functools
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map_, _broadcast_to_and_flatten, TreeSpec
from functools import partial
import os
import itertools
from torch._C._functorch import _add_batch_dim, _remove_batch_dim, _vmap_decrement_nesting, _vmap_increment_nesting, is_batchedtensor
in_dims_t = Union[int, Tuple]
out_dims_t = Union[int, Tuple[int, ...]]

def doesnt_support_saved_tensors_hooks(f):
    if False:
        for i in range(10):
            print('nop')
    message = "torch.func transforms don't yet support saved tensor hooks. Please open an issue with your use case."

    @functools.wraps(f)
    def fn(*args, **kwargs):
        if False:
            print('Hello World!')
        with torch.autograd.graph.disable_saved_tensors_hooks(message):
            return f(*args, **kwargs)
    return fn

def _validate_and_get_batch_size(flat_in_dims: List[Optional[int]], flat_args: List) -> int:
    if False:
        while True:
            i = 10
    batch_sizes = [arg.size(in_dim) for (in_dim, arg) in zip(flat_in_dims, flat_args) if in_dim is not None]
    if len(batch_sizes) == 0:
        raise ValueError('vmap: Expected at least one Tensor to vmap over')
    if batch_sizes and any((size != batch_sizes[0] for size in batch_sizes)):
        raise ValueError(f'vmap: Expected all tensors to have the same size in the mapped dimension, got sizes {batch_sizes} for the mapped dimension')
    return batch_sizes[0]

def _num_outputs(batched_outputs: Union[Tensor, Tuple[Tensor, ...]]) -> int:
    if False:
        print('Hello World!')
    if isinstance(batched_outputs, tuple):
        return len(batched_outputs)
    return 1

def _as_tuple(value: Any, num_elements: int, error_message_lambda: Callable[[], str]) -> Tuple:
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(value, tuple):
        return (value,) * num_elements
    if len(value) != num_elements:
        raise ValueError(error_message_lambda())
    return value

def _process_batched_inputs(in_dims: in_dims_t, args: Tuple, func: Callable) -> Tuple[int, List[Any], List[Any], TreeSpec]:
    if False:
        i = 10
        return i + 15
    if not isinstance(in_dims, int) and (not isinstance(in_dims, tuple)):
        raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): expected `in_dims` to be int or a (potentially nested) tuple matching the structure of inputs, got: {type(in_dims)}.')
    if len(args) == 0:
        raise ValueError(f'vmap({_get_name(func)})(<inputs>): got no inputs. Maybe you forgot to add inputs, or you are trying to vmap over a function with no inputs. The latter is unsupported.')
    (flat_args, args_spec) = tree_flatten(args)
    flat_in_dims = _broadcast_to_and_flatten(in_dims, args_spec)
    if flat_in_dims is None:
        raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): in_dims is not compatible with the structure of `inputs`. in_dims has structure {tree_flatten(in_dims)[1]} but inputs has structure {args_spec}.')
    for (i, (arg, in_dim)) in enumerate(zip(flat_args, flat_in_dims)):
        if not isinstance(in_dim, int) and in_dim is not None:
            raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): Got in_dim={in_dim} for an input but in_dim must be either an integer dimension or None.')
        if isinstance(in_dim, int) and (not isinstance(arg, Tensor)):
            raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): Got in_dim={in_dim} for an input but the input is of type {type(arg)}. We cannot vmap over non-Tensor arguments, please use None as the respective in_dim')
        if in_dim is not None and (in_dim < -arg.dim() or in_dim >= arg.dim()):
            raise ValueError(f'vmap({_get_name(func)}, in_dims={in_dims}, ...)(<inputs>): Got in_dim={in_dim} for some input, but that input is a Tensor of dimensionality {arg.dim()} so expected in_dim to satisfy -{arg.dim()} <= in_dim < {arg.dim()}.')
        if in_dim is not None and in_dim < 0:
            flat_in_dims[i] = in_dim % arg.dim()
    return (_validate_and_get_batch_size(flat_in_dims, flat_args), flat_in_dims, flat_args, args_spec)

def _create_batched_inputs(flat_in_dims: List[Any], flat_args: List[Any], vmap_level: int, args_spec) -> Tuple:
    if False:
        print('Hello World!')
    batched_inputs = [arg if in_dim is None else _add_batch_dim(arg, in_dim, vmap_level) for (in_dim, arg) in zip(flat_in_dims, flat_args)]
    return tree_unflatten(batched_inputs, args_spec)

def _maybe_remove_batch_dim(name, batched_output, vmap_level, batch_size, out_dim):
    if False:
        for i in range(10):
            print('nop')
    if out_dim is None:
        if isinstance(batched_output, torch.Tensor) and is_batchedtensor(batched_output):
            raise ValueError(f'vmap({name}, ...): `{name}` can not return a BatchedTensor when out_dim is None')
        return batched_output
    if not isinstance(batched_output, torch.Tensor):
        raise ValueError(f'vmap({name}, ...): `{name}` must only return Tensors, got type {type(batched_output)}. Did you mean to set out_dim= to None for output?')
    return _remove_batch_dim(batched_output, vmap_level, batch_size, out_dim)

def _unwrap_batched(batched_outputs: Union[Tensor, Tuple[Tensor, ...]], out_dims: out_dims_t, vmap_level: int, batch_size: int, func: Callable) -> Tuple:
    if False:
        print('Hello World!')
    (flat_batched_outputs, output_spec) = tree_flatten(batched_outputs)

    def incompatible_error():
        if False:
            i = 10
            return i + 15
        raise ValueError(f'vmap({_get_name(func)}, ..., out_dims={out_dims})(<inputs>): out_dims is not compatible with the structure of `outputs`. out_dims has structure {tree_flatten(out_dims)[1]} but outputs has structure {output_spec}.')
    if isinstance(batched_outputs, torch.Tensor):
        if isinstance(out_dims, int):
            flat_out_dims = [out_dims]
        elif isinstance(out_dims, tuple) and len(out_dims) == 1:
            flat_out_dims = out_dims
        elif out_dims is None:
            flat_out_dims = [out_dims]
        else:
            incompatible_error()
    else:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, output_spec)
        if flat_out_dims is None:
            incompatible_error()
    flat_outputs = [_maybe_remove_batch_dim(_get_name(func), batched_output, vmap_level, batch_size, out_dim) for (batched_output, out_dim) in zip(flat_batched_outputs, flat_out_dims)]
    return tree_unflatten(flat_outputs, output_spec)

def _check_int_or_none(x, func, out_dims):
    if False:
        for i in range(10):
            print('nop')
    if isinstance(x, int):
        return
    if x is None:
        return
    raise ValueError(f'vmap({_get_name(func)}, ..., out_dims={out_dims}): `out_dims` must be an int, None or a python collection of ints representing where in the outputs the vmapped dimension should appear.')

def _check_out_dims_is_int_or_int_pytree(out_dims: out_dims_t, func: Callable) -> None:
    if False:
        for i in range(10):
            print('nop')
    if isinstance(out_dims, int):
        return
    tree_map_(partial(_check_int_or_none, func=func, out_dims=out_dims), out_dims)

def _get_name(func: Callable):
    if False:
        while True:
            i = 10
    if hasattr(func, '__name__'):
        return func.__name__
    return repr(func)
DECOMPOSITIONS_LOADED = False
VMAP_DECOMPOSITIONS_LIB = None

def lazy_load_decompositions():
    if False:
        return 10
    global DECOMPOSITIONS_LOADED
    if DECOMPOSITIONS_LOADED:
        return
    DECOMPOSITIONS_LOADED = True
    if not (os.environ.get('PYTORCH_JIT', '1') == '1' and __debug__):
        return
    global VMAP_DECOMPOSITIONS_LIB
    VMAP_DECOMPOSITIONS_LIB = torch.library.Library('aten', 'IMPL', 'FuncTorchBatched')
    from torch._decomp import decomposition_table

    def _register_python_decomposition_vmap(decomp):
        if False:
            while True:
                i = 10
        if decomp in decomposition_table:
            VMAP_DECOMPOSITIONS_LIB.impl(decomp, decomposition_table[decomp])
        else:
            raise RuntimeError(f'could not find decomposition for {decomp}')
    _register_python_decomposition_vmap(torch.ops.aten.mse_loss_backward.default)
    _register_python_decomposition_vmap(torch.ops.aten.smooth_l1_loss_backward.default)
    _register_python_decomposition_vmap(torch.ops.aten.huber_loss_backward.default)
    _register_python_decomposition_vmap(torch.ops.aten.nll_loss_forward.default)
    _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_forward.default)
    _register_python_decomposition_vmap(torch.ops.aten.nll_loss_backward.default)
    _register_python_decomposition_vmap(torch.ops.aten.nll_loss2d_backward.default)
    _register_python_decomposition_vmap(torch.ops.aten.addr.default)

def vmap_impl(func, in_dims, out_dims, randomness, chunk_size, *args, **kwargs):
    if False:
        return 10
    lazy_load_decompositions()
    _check_out_dims_is_int_or_int_pytree(out_dims, func)
    (batch_size, flat_in_dims, flat_args, args_spec) = _process_batched_inputs(in_dims, args, func)
    if chunk_size is not None:
        chunks_flat_args = _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size)
        return _chunked_vmap(func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs)
    return _flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs)

def get_chunk_sizes(total_elems, chunk_size):
    if False:
        while True:
            i = 10
    n_chunks = n_chunks = total_elems // chunk_size
    chunk_sizes = [chunk_size] * n_chunks
    remainder = total_elems % chunk_size
    if remainder != 0:
        chunk_sizes.append(remainder)
    return chunk_sizes

def _get_chunked_inputs(flat_args, flat_in_dims, batch_size, chunk_size):
    if False:
        while True:
            i = 10
    split_idxs = (batch_size,)
    if chunk_size is not None:
        chunk_sizes = get_chunk_sizes(batch_size, chunk_size)
        split_idxs = tuple(itertools.accumulate(chunk_sizes))
    flat_args_chunks = tuple((t.tensor_split(split_idxs, dim=in_dim) if in_dim is not None else [t] * len(split_idxs) for (t, in_dim) in zip(flat_args, flat_in_dims)))
    chunks_flat_args = zip(*flat_args_chunks)
    return chunks_flat_args

def _flatten_chunks_output(chunks_output_):
    if False:
        return 10
    flat_chunks_output = []
    arg_spec = None
    for output in chunks_output_:
        (flat_output, arg_specs) = tree_flatten(output)
        flat_chunks_output.append(flat_output)
        if arg_spec is None:
            arg_spec = arg_specs
    flat_output_chunks = list(zip(*flat_chunks_output))
    return (flat_output_chunks, arg_spec)

def _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks):
    if False:
        return 10
    flat_out_dims = _broadcast_to_and_flatten(out_dims, arg_spec)
    assert len(flat_out_dims) == len(flat_output_chunks)
    flat_output = []
    for (idx, out_dim) in enumerate(flat_out_dims):
        flat_output.append(torch.cat(flat_output_chunks[idx], dim=out_dim))
        flat_output_chunks[idx] = None
    return flat_output

def _chunked_vmap(func, flat_in_dims, chunks_flat_args, args_spec, out_dims, randomness, **kwargs):
    if False:
        while True:
            i = 10
    chunks_output = []
    rs = torch.get_rng_state() if randomness == 'same' else None
    for flat_args in chunks_flat_args:
        batch_size = _validate_and_get_batch_size(flat_in_dims, flat_args)
        if batch_size == 0:
            continue
        if rs is not None:
            torch.set_rng_state(rs)
        chunks_output.append(_flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs))
    (flat_output_chunks, arg_spec) = _flatten_chunks_output(chunks_output)
    del chunks_output
    flat_output = _concat_chunked_outputs(out_dims, arg_spec, flat_output_chunks)
    return tree_unflatten(flat_output, arg_spec)

def _check_randomness_arg(randomness):
    if False:
        return 10
    if randomness not in ['error', 'different', 'same']:
        raise RuntimeError(f"Only allowed values for randomness are 'error', 'different', or 'same'. Got {randomness}")

@doesnt_support_saved_tensors_hooks
def _flat_vmap(func, batch_size, flat_in_dims, flat_args, args_spec, out_dims, randomness, **kwargs):
    if False:
        print('Hello World!')
    vmap_level = _vmap_increment_nesting(batch_size, randomness)
    try:
        batched_inputs = _create_batched_inputs(flat_in_dims, flat_args, vmap_level, args_spec)
        batched_outputs = func(*batched_inputs, **kwargs)
        return _unwrap_batched(batched_outputs, out_dims, vmap_level, batch_size, func)
    finally:
        _vmap_decrement_nesting()

@doesnt_support_saved_tensors_hooks
def restore_vmap(func, in_dims, batch_size, randomness):
    if False:
        while True:
            i = 10

    def inner(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        vmap_level = _vmap_increment_nesting(batch_size, randomness)
        try:
            batched_inputs = wrap_batched(args, in_dims, vmap_level)
            batched_outputs = func(*batched_inputs, **kwargs)
            return unwrap_batched(batched_outputs, vmap_level)
        finally:
            _vmap_decrement_nesting()
    return inner

def wrap_batched(args, bdims, level):
    if False:
        while True:
            i = 10
    (flat_args, spec) = tree_flatten(args)
    flat_bdims = _broadcast_to_and_flatten(bdims, spec)
    assert flat_bdims is not None
    result = _create_batched_inputs(flat_bdims, flat_args, level, spec)
    return result

def unwrap_batched(args, level):
    if False:
        for i in range(10):
            print('nop')
    (flat_args, spec) = tree_flatten(args)
    if len(flat_args) == 0:
        return (args, ())
    result = [torch._C._functorch._unwrap_batched(arg, level) if isinstance(arg, torch.Tensor) else (arg, None) for arg in flat_args]
    (output, bdims) = zip(*result)
    return (tree_unflatten(output, spec), tree_unflatten(bdims, spec))