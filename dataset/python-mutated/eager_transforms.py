from typing import Callable, Union, Tuple, List, Any, Optional
import torch
from functools import partial, wraps
import contextlib
from torch.utils._pytree import tree_flatten, tree_unflatten, tree_map, tree_map_only, tree_map_, treespec_pprint
from torch.utils import _pytree as pytree
from torch.fx.experimental import const_fold
from torch.fx.experimental.proxy_tensor import make_fx
import torch.autograd.forward_ad as fwAD
from torch._subclasses.functional_tensor import FunctionalTensor
from .vmap import doesnt_support_saved_tensors_hooks, get_chunk_sizes
from .apis import vmap
from torch._C._functorch import _wrap_for_grad, _unwrap_for_grad, _grad_increment_nesting, _grad_decrement_nesting, _jvp_increment_nesting, _jvp_decrement_nesting, _wrap_functional_tensor, _unwrap_functional_tensor, _func_decrement_nesting, _func_increment_nesting, _assert_wrapped_functional, _propagate_functional_input_mutation, set_inplace_requires_grad_allowed, get_inplace_requires_grad_allowed
from torch._functorch.utils import exposed_in, argnums_t

def lazy_dynamo_disable(func):
    if False:
        return 10
    import torch._dynamo
    return torch._dynamo.disable(func)

@contextlib.contextmanager
def enable_inplace_requires_grad(enabled=True):
    if False:
        return 10
    prev_state = get_inplace_requires_grad_allowed()
    set_inplace_requires_grad_allowed(enabled)
    try:
        yield
    finally:
        set_inplace_requires_grad_allowed(prev_state)

def _create_differentiable(inps, level=None):
    if False:
        print('Hello World!')

    def create_differentiable(x):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(x, torch.Tensor):
            with enable_inplace_requires_grad():
                return x.requires_grad_()
        raise ValueError(f'Thing passed to transform API must be Tensor, got {type(x)}')
    return tree_map(create_differentiable, inps)

def _undo_create_differentiable(inps, level=None):
    if False:
        i = 10
        return i + 15

    def unwrap_tensors(x):
        if False:
            i = 10
            return i + 15
        if isinstance(x, torch.Tensor):
            return _unwrap_for_grad(x, level)
        if isinstance(x, tuple):
            return tree_map(unwrap_tensors, tuple(x))
        raise RuntimeError(f'Expected tensors, got unsupported type {type(x)}')
    return tree_map(unwrap_tensors, inps)

def _is_differentiable(maybe_tensor):
    if False:
        print('Hello World!')
    if not isinstance(maybe_tensor, torch.Tensor):
        return False
    return maybe_tensor.requires_grad

def _any_differentiable(tensor_or_tuple_of_tensors):
    if False:
        while True:
            i = 10
    (flat_args, _) = tree_unflatten(tensor_or_tuple_of_tensors)
    return any(tuple(map(_is_differentiable, flat_args)))

def _wrap_tensor_for_grad(maybe_tensor, level):
    if False:
        while True:
            i = 10
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    return _wrap_for_grad(maybe_tensor, level)

def _wrap_all_tensors(tensor_pytree, level):
    if False:
        for i in range(10):
            print('nop')
    return tree_map(partial(_wrap_tensor_for_grad, level=level), tensor_pytree)

def _as_tuple(val):
    if False:
        return 10
    if isinstance(val, tuple):
        return val
    return (val,)

def _autograd_grad(outputs, inputs, grad_outputs=None, retain_graph=False, create_graph=True):
    if False:
        print('Hello World!')
    if grad_outputs is None:
        diff_outputs = tuple((out for out in outputs if out.requires_grad))
    else:
        result = tuple(((out, go) for (out, go) in zip(outputs, grad_outputs) if out.requires_grad))
        if len(result) == 0:
            (diff_outputs, grad_outputs) = ((), ())
        else:
            (diff_outputs, grad_outputs) = zip(*result)
    if len(diff_outputs) == 0:
        return tuple((torch.zeros_like(inp) for inp in inputs))
    grad_inputs = torch.autograd.grad(diff_outputs, inputs, grad_outputs, retain_graph=retain_graph, create_graph=create_graph, allow_unused=True)
    grad_inputs = tuple((torch.zeros_like(inp) if gi is None else gi for (gi, inp) in zip(grad_inputs, inputs)))
    return grad_inputs

@exposed_in('torch.func')
def vjp(func: Callable, *primals, has_aux: bool=False):
    if False:
        while True:
            i = 10
    '\n    Standing for the vector-Jacobian product, returns a tuple containing the\n    results of ``func`` applied to ``primals`` and a function that, when\n    given ``cotangents``, computes the reverse-mode Jacobian of ``func`` with\n    respect to ``primals`` times ``cotangents``.\n\n    Args:\n        func (Callable): A Python function that takes one or more arguments. Must\n            return one or more Tensors.\n        primals (Tensors): Positional arguments to ``func`` that must all be\n            Tensors. The returned function will also be computing the\n            derivative with respect to these arguments\n        has_aux (bool): Flag indicating that ``func`` returns a\n            ``(output, aux)`` tuple where the first element is the output of\n            the function to be differentiated and the second element is\n            other auxiliary objects that will not be differentiated.\n            Default: False.\n\n    Returns:\n        Returns a ``(output, vjp_fn)`` tuple containing the output of ``func``\n        applied to ``primals`` and a function that computes the vjp of\n        ``func`` with respect to all ``primals`` using the cotangents passed\n        to the returned function. If ``has_aux is True``, then instead returns a\n        ``(output, vjp_fn, aux)`` tuple.\n        The returned ``vjp_fn`` function will return a tuple of each VJP.\n\n    When used in simple cases, :func:`vjp` behaves the same as :func:`grad`\n\n        >>> x = torch.randn([5])\n        >>> f = lambda x: x.sin().sum()\n        >>> (_, vjpfunc) = torch.func.vjp(f, x)\n        >>> grad = vjpfunc(torch.tensor(1.))[0]\n        >>> assert torch.allclose(grad, torch.func.grad(f)(x))\n\n    However, :func:`vjp` can support functions with multiple outputs by\n    passing in the cotangents for each of the outputs\n\n        >>> x = torch.randn([5])\n        >>> f = lambda x: (x.sin(), x.cos())\n        >>> (_, vjpfunc) = torch.func.vjp(f, x)\n        >>> vjps = vjpfunc((torch.ones([5]), torch.ones([5])))\n        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())\n\n    :func:`vjp` can even support outputs being Python structs\n\n        >>> x = torch.randn([5])\n        >>> f = lambda x: {\'first\': x.sin(), \'second\': x.cos()}\n        >>> (_, vjpfunc) = torch.func.vjp(f, x)\n        >>> cotangents = {\'first\': torch.ones([5]), \'second\': torch.ones([5])}\n        >>> vjps = vjpfunc(cotangents)\n        >>> assert torch.allclose(vjps[0], x.cos() + -x.sin())\n\n    The function returned by :func:`vjp` will compute the partials with\n    respect to each of the ``primals``\n\n        >>> x, y = torch.randn([5, 4]), torch.randn([4, 5])\n        >>> (_, vjpfunc) = torch.func.vjp(torch.matmul, x, y)\n        >>> cotangents = torch.randn([5, 5])\n        >>> vjps = vjpfunc(cotangents)\n        >>> assert len(vjps) == 2\n        >>> assert torch.allclose(vjps[0], torch.matmul(cotangents, y.transpose(0, 1)))\n        >>> assert torch.allclose(vjps[1], torch.matmul(x.transpose(0, 1), cotangents))\n\n    ``primals`` are the positional arguments for ``f``. All kwargs use their\n    default value\n\n        >>> x = torch.randn([5])\n        >>> def f(x, scale=4.):\n        >>>   return x * scale\n        >>>\n        >>> (_, vjpfunc) = torch.func.vjp(f, x)\n        >>> vjps = vjpfunc(torch.ones_like(x))\n        >>> assert torch.allclose(vjps[0], torch.full(x.shape, 4.))\n\n    .. note::\n        Using PyTorch ``torch.no_grad`` together with ``vjp``.\n        Case 1: Using ``torch.no_grad`` inside a function:\n\n            >>> def f(x):\n            >>>     with torch.no_grad():\n            >>>         c = x ** 2\n            >>>     return x - c\n\n        In this case, ``vjp(f)(x)`` will respect the inner ``torch.no_grad``.\n\n        Case 2: Using ``vjp`` inside ``torch.no_grad`` context manager:\n\n            >>> # xdoctest: +SKIP(failing)\n            >>> with torch.no_grad():\n            >>>     vjp(f)(x)\n\n        In this case, ``vjp`` will respect the inner ``torch.no_grad``, but not the\n        outer one. This is because ``vjp`` is a "function transform": its result\n        should not depend on the result of a context manager outside of ``f``.\n    '
    return _vjp_with_argnums(func, *primals, has_aux=has_aux)

@doesnt_support_saved_tensors_hooks
def _vjp_with_argnums(func: Callable, *primals, argnums: Optional[argnums_t]=None, has_aux: bool=False):
    if False:
        print('Hello World!')
    level = _grad_increment_nesting()
    try:
        with torch.enable_grad():
            primals = _wrap_all_tensors(primals, level)
            if argnums is None:
                diff_primals = _create_differentiable(primals, level)
            else:
                diff_primals = _slice_argnums(primals, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_primals)
            primals_out = func(*primals)
            if has_aux:
                if not (isinstance(primals_out, tuple) and len(primals_out) == 2):
                    raise RuntimeError('vjp(f, *primals): output of function f should be a tuple: (output, aux) if has_aux is True')
                (primals_out, aux) = primals_out
                aux = _undo_create_differentiable(aux, level)
            (flat_primals_out, primals_out_spec) = tree_flatten(primals_out)
            assert_non_empty_tensor_output(flat_primals_out, 'vjp(f, *primals)')
            (flat_diff_primals, primals_spec) = tree_flatten(diff_primals)
            results = _undo_create_differentiable(primals_out, level)
            for primal_out in flat_primals_out:
                assert isinstance(primal_out, torch.Tensor)
                if primal_out.is_floating_point() or primal_out.is_complex():
                    continue
                raise RuntimeError(f'vjp(f, ...): All outputs of f must be floating-point or complex Tensors, got Tensor with dtype {primal_out.dtype}')

        def wrapper(cotangents, retain_graph=True, create_graph=None):
            if False:
                return 10
            if create_graph is None:
                create_graph = torch.is_grad_enabled()
            (flat_cotangents, cotangents_spec) = tree_flatten(cotangents)
            if primals_out_spec != cotangents_spec:
                raise RuntimeError(f'Expected pytree structure of cotangents to be the same as pytree structure of outputs to the function. cotangents: {treespec_pprint(cotangents_spec)}, primal output: {treespec_pprint(primals_out_spec)}')
            result = _autograd_grad(flat_primals_out, flat_diff_primals, flat_cotangents, retain_graph=retain_graph, create_graph=create_graph)
            return tree_unflatten(result, primals_spec)
    finally:
        _grad_decrement_nesting()
    if has_aux:
        return (results, wrapper, aux)
    else:
        return (results, wrapper)

def _safe_zero_index(x):
    if False:
        while True:
            i = 10
    assert len(x) == 1
    return x[0]

def error_if_complex(func_name, args, is_input):
    if False:
        return 10
    flat_args = pytree.tree_leaves(args)
    for (idx, arg) in enumerate(flat_args):
        if isinstance(arg, torch.Tensor) and arg.dtype.is_complex:
            input_or_output = 'inputs' if is_input else 'outputs'
            err_msg = f'{func_name}: Expected all {input_or_output} to be real but received complex tensor at flattened input idx: {idx}'
            raise RuntimeError(err_msg)

@exposed_in('torch.func')
def jacrev(func: Callable, argnums: Union[int, Tuple[int]]=0, *, has_aux=False, chunk_size: Optional[int]=None, _preallocate_and_copy=False):
    if False:
        print('Hello World!')
    '\n    Computes the Jacobian of ``func`` with respect to the arg(s) at index\n    ``argnum`` using reverse mode autodiff\n\n    .. note::\n        Using :attr:`chunk_size=1` is equivalent to computing the jacobian\n        row-by-row with a for-loop i.e. the constraints of :func:`vmap` are\n        not applicable.\n\n    Args:\n        func (function): A Python function that takes one or more arguments,\n            one of which must be a Tensor, and returns one or more Tensors\n        argnums (int or Tuple[int]): Optional, integer or tuple of integers,\n            saying which arguments to get the Jacobian with respect to.\n            Default: 0.\n        has_aux (bool): Flag indicating that ``func`` returns a\n            ``(output, aux)`` tuple where the first element is the output of\n            the function to be differentiated and the second element is\n            auxiliary objects that will not be differentiated.\n            Default: False.\n        chunk_size (None or int): If None (default), use the maximum chunk size\n            (equivalent to doing a single vmap over vjp to compute the jacobian).\n            If 1, then compute the jacobian row-by-row with a for-loop.\n            If not None, then compute the jacobian :attr:`chunk_size` rows at a time\n            (equivalent to doing multiple vmap over vjp). If you run into memory issues computing\n            the jacobian, please try to specify a non-None chunk_size.\n\n    Returns:\n        Returns a function that takes in the same inputs as ``func`` and\n        returns the Jacobian of ``func`` with respect to the arg(s) at\n        ``argnums``. If ``has_aux is True``, then the returned function\n        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``\n        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.\n\n    A basic usage with a pointwise, unary operation will give a diagonal array\n    as the Jacobian\n\n        >>> from torch.func import jacrev\n        >>> x = torch.randn(5)\n        >>> jacobian = jacrev(torch.sin)(x)\n        >>> expected = torch.diag(torch.cos(x))\n        >>> assert torch.allclose(jacobian, expected)\n\n    If you would like to compute the output of the function as well as the\n    jacobian of the function, use the ``has_aux`` flag to return the output\n    as an auxiliary object:\n\n        >>> from torch.func import jacrev\n        >>> x = torch.randn(5)\n        >>>\n        >>> def f(x):\n        >>>   return x.sin()\n        >>>\n        >>> def g(x):\n        >>>   result = f(x)\n        >>>   return result, result\n        >>>\n        >>> jacobian_f, f_x = jacrev(g, has_aux=True)(x)\n        >>> assert torch.allclose(f_x, f(x))\n\n    :func:`jacrev` can be composed with vmap to produce batched\n    Jacobians:\n\n        >>> from torch.func import jacrev, vmap\n        >>> x = torch.randn(64, 5)\n        >>> jacobian = vmap(jacrev(torch.sin))(x)\n        >>> assert jacobian.shape == (64, 5, 5)\n\n    Additionally, :func:`jacrev` can be composed with itself to produce\n    Hessians\n\n        >>> from torch.func import jacrev\n        >>> def f(x):\n        >>>   return x.sin().sum()\n        >>>\n        >>> x = torch.randn(5)\n        >>> hessian = jacrev(jacrev(f))(x)\n        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))\n\n    By default, :func:`jacrev` computes the Jacobian with respect to the first\n    input. However, it can compute the Jacboian with respect to a different\n    argument by using ``argnums``:\n\n        >>> from torch.func import jacrev\n        >>> def f(x, y):\n        >>>   return x + y ** 2\n        >>>\n        >>> x, y = torch.randn(5), torch.randn(5)\n        >>> jacobian = jacrev(f, argnums=1)(x, y)\n        >>> expected = torch.diag(2 * y)\n        >>> assert torch.allclose(jacobian, expected)\n\n    Additionally, passing a tuple to ``argnums`` will compute the Jacobian\n    with respect to multiple arguments\n\n        >>> from torch.func import jacrev\n        >>> def f(x, y):\n        >>>   return x + y ** 2\n        >>>\n        >>> x, y = torch.randn(5), torch.randn(5)\n        >>> jacobian = jacrev(f, argnums=(0, 1))(x, y)\n        >>> expectedX = torch.diag(torch.ones_like(x))\n        >>> expectedY = torch.diag(2 * y)\n        >>> assert torch.allclose(jacobian[0], expectedX)\n        >>> assert torch.allclose(jacobian[1], expectedY)\n\n    .. note::\n        Using PyTorch ``torch.no_grad`` together with ``jacrev``.\n        Case 1: Using ``torch.no_grad`` inside a function:\n\n            >>> def f(x):\n            >>>     with torch.no_grad():\n            >>>         c = x ** 2\n            >>>     return x - c\n\n        In this case, ``jacrev(f)(x)`` will respect the inner ``torch.no_grad``.\n\n        Case 2: Using ``jacrev`` inside ``torch.no_grad`` context manager:\n\n            >>> with torch.no_grad():\n            >>>     jacrev(f)(x)\n\n        In this case, ``jacrev`` will respect the inner ``torch.no_grad``, but not the\n        outer one. This is because ``jacrev`` is a "function transform": its result\n        should not depend on the result of a context manager outside of ``f``.\n    '
    if not (chunk_size is None or chunk_size > 0):
        raise ValueError('jacrev: `chunk_size` should be greater than 0.')

    @wraps(func)
    def wrapper_fn(*args):
        if False:
            print('Hello World!')
        error_if_complex('jacrev', args, is_input=True)
        vjp_out = _vjp_with_argnums(func, *args, argnums=argnums, has_aux=has_aux)
        if has_aux:
            (output, vjp_fn, aux) = vjp_out
        else:
            (output, vjp_fn) = vjp_out
        (flat_output, output_spec) = tree_flatten(output)
        error_if_complex('jacrev', flat_output, is_input=False)
        flat_output_numels = tuple((out.numel() for out in flat_output))
        primals = _slice_argnums(args, argnums)
        (flat_primals, primals_spec) = tree_flatten(primals)

        def compute_jacobian_stacked():
            if False:
                print('Hello World!')
            chunked_results = []
            for flat_basis_chunk in _chunked_standard_basis_for_(flat_output, flat_output_numels, chunk_size=chunk_size):
                if chunk_size == 1:
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1
                    flat_basis_chunk = tree_map(lambda t: torch.squeeze(t, 0), flat_basis_chunk)
                basis = tree_unflatten(flat_basis_chunk, output_spec)
                if chunk_size == 1:
                    chunked_result = vjp_fn(basis)
                else:
                    chunked_result = vmap(vjp_fn)(basis)
                flat_results = pytree.tree_leaves(chunked_result)
                if chunk_size == 1:
                    flat_results = tree_map(lambda t: torch.unsqueeze(t, 0), flat_results)
                chunked_results.append(flat_results)
            if len(chunked_results) == 1:
                return chunked_results[0]
            flat_results = []
            for idx in range(len(flat_primals)):
                r = tuple((r_[idx] for r_ in chunked_results))
                flat_results.append(torch.cat(r, 0))
            return flat_results

        def compute_jacobian_preallocate_and_copy():
            if False:
                print('Hello World!')
            out_vec_size = sum(flat_output_numels)
            if not (chunk_size is None or chunk_size >= out_vec_size):
                stacked_results = [primal.new_zeros(out_vec_size, *primal.shape) for primal in flat_primals]
            for (idx, flat_basis_chunk) in enumerate(_chunked_standard_basis_for_(flat_output, flat_output_numels, chunk_size=chunk_size)):
                if chunk_size == 1:
                    for t in flat_basis_chunk:
                        assert t.size(0) == 1
                    flat_basis_chunk = [torch.squeeze(t, 0) for t in flat_basis_chunk]
                basis = tree_unflatten(flat_basis_chunk, output_spec)
                if chunk_size == 1:
                    chunked_result = vjp_fn(basis)
                else:
                    chunked_result = vmap(vjp_fn)(basis)
                flat_results = pytree.tree_leaves(chunked_result)
                if chunk_size is None or chunk_size >= out_vec_size:
                    if chunk_size == 1:
                        flat_results = tree_map(lambda t: torch.unsqueeze(t, 0), flat_results)
                    return flat_results
                for (r, sr) in zip(flat_results, stacked_results):
                    sr[idx * chunk_size:(idx + 1) * chunk_size].copy_(r)
            return stacked_results
        if _preallocate_and_copy:
            flat_jacobians_per_input = compute_jacobian_preallocate_and_copy()
        else:
            flat_jacobians_per_input = compute_jacobian_stacked()
        flat_jacobians_per_input = [result.split(flat_output_numels, dim=0) for result in flat_jacobians_per_input]
        flat_input_flat_output = [tuple((split.view(out.shape + primal.shape) for (split, out) in zip(splits, flat_output))) for (splits, primal) in zip(flat_jacobians_per_input, flat_primals)]
        flat_output_flat_input = tuple(zip(*flat_input_flat_output))
        flat_output_input = tuple((tree_unflatten(flat_input, primals_spec) for flat_input in flat_output_flat_input))
        if isinstance(argnums, int):
            flat_output_input = tuple((_safe_zero_index(flat_input) for flat_input in flat_output_input))
        output_input = tree_unflatten(flat_output_input, output_spec)
        if has_aux:
            return (output_input, aux)
        return output_input
    return wrapper_fn

def _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
    if False:
        print('Hello World!')
    assert len(tensors) == len(tensor_numels)
    assert len(tensors) > 0
    assert chunk_size is None or chunk_size > 0
    total_numel = sum(tensor_numels)
    if chunk_size and chunk_size < total_numel:
        chunk_numels = get_chunk_sizes(total_numel, chunk_size)
    else:
        chunk_size = total_numel
        chunk_numels = [total_numel]
    diag_start_indices = (0, *torch.tensor(tensor_numels).cumsum(dim=0)[:-1].neg().unbind())
    for (chunk_idx, total_numel) in enumerate(chunk_numels):
        chunks = tuple((tensor.new_zeros(total_numel, tensor_numel) for (tensor, tensor_numel) in zip(tensors, tensor_numels)))
        for (chunk, diag_start_idx) in zip(chunks, diag_start_indices):
            chunk.diagonal(diag_start_idx + chunk_idx * chunk_size).fill_(1)
        chunks = tuple((chunk.view(total_numel, *tensor.shape) for (chunk, tensor) in zip(chunks, tensors)))
        yield chunks

def _construct_standard_basis_for(tensors, tensor_numels):
    if False:
        return 10
    for basis in _chunked_standard_basis_for_(tensors, tensor_numels, chunk_size=None):
        return basis

def _validate_and_wrap_argnum(argnum, num_args):
    if False:
        return 10
    if not isinstance(argnum, int):
        raise RuntimeError(f'argnum must be int, got: {type(argnum)}')
    if argnum >= 0 and argnum < num_args:
        return argnum
    if argnum < 0 and argnum >= -num_args:
        return argnum + num_args
    raise RuntimeError(f'Got argnum={argnum}, but only {num_args} positional inputs')

def _check_unique_non_empty(argnums):
    if False:
        i = 10
        return i + 15
    if isinstance(argnums, tuple):
        if len(argnums) == 0:
            raise RuntimeError('argnums must be non-empty')
        if len(set(argnums)) != len(argnums):
            raise RuntimeError(f'argnums elements must be unique, got {argnums}')

def _replace_args(old_args, new_args, argnums):
    if False:
        while True:
            i = 10
    if isinstance(argnums, int):
        if len(new_args) != 1:
            raise RuntimeError(f'new_args should be of size 1, was of size {len(new_args)}')
        return tuple((new_args[0] if i == argnums else old_args[i] for i in range(len(old_args))))
    if isinstance(argnums, tuple):
        if len(new_args) != len(argnums):
            raise RuntimeError(f'new_args should have the same size as argnums. Argnums size {len(argnums)}, new_args size {len(new_args)}')

        def get_right_elem(i):
            if False:
                print('Hello World!')
            return new_args[argnums.index(i)] if i in argnums else old_args[i]
        return tuple((get_right_elem(i) for i in range(len(old_args))))
    raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')

def _validate_and_wrap_argnums(argnums, num_args):
    if False:
        i = 10
        return i + 15
    if isinstance(argnums, int):
        return _validate_and_wrap_argnum(argnums, num_args)
    if isinstance(argnums, tuple):
        return tuple((_validate_and_wrap_argnum(argnum, num_args) for argnum in argnums))
    raise AssertionError('Should never get here')

def _slice_argnums(args, argnums, as_tuple=True):
    if False:
        while True:
            i = 10
    if not isinstance(argnums, int) and (not isinstance(argnums, tuple)):
        raise RuntimeError(f'argnums must be int or Tuple[int, ...], got: {type(argnums)}')
    argnums = _validate_and_wrap_argnums(argnums, len(args))
    _check_unique_non_empty(argnums)
    if isinstance(argnums, int):
        if as_tuple:
            return (args[argnums],)
        else:
            return args[argnums]
    return tuple((args[i] for i in argnums))
JVP_NESTING = 0

@contextlib.contextmanager
def noop():
    if False:
        return 10
    yield

def assert_flat_tuple_of_tensors(elts: Any, api: str, argname: str) -> None:
    if False:
        print('Hello World!')
    if not isinstance(elts, tuple):
        raise RuntimeError(f'{api}: Expected {argname} to be a tuple of Tensors, got {type(elts)}')
    for elt in elts:
        if isinstance(elt, torch.Tensor):
            continue
        raise RuntimeError(f'{api}: Expected {argname} to be a tuple of Tensors, got a tuple with an element of type {type(elt)}')
    if len(elts) == 0:
        raise RuntimeError(f'{api}: Expected {argname} to be a non-empty tuple of Tensors.')

def assert_non_empty_tensor_output(output: List[Any], api: str) -> None:
    if False:
        i = 10
        return i + 15
    if output == [None] or len(output) < 1:
        raise RuntimeError(f'{api}: Expected f to be a function that has non-empty output (got output = {output})')
    for o in output:
        if not isinstance(o, torch.Tensor):
            raise RuntimeError(f'{api}: expected f(*primals) to return only tensors, got unsupported type {type(o)}')

def assert_output_is_tensor_or_tensors(output: Any, api: str) -> None:
    if False:
        return 10
    if isinstance(output, torch.Tensor):
        return
    if not isinstance(output, tuple):
        raise RuntimeError(f'{api}: Expected output of f to be a Tensor or Tensors, got {type(output)}')
    if len(output) == 0:
        raise RuntimeError(f'{api}: Expected output of f to be a non-empty tuple of Tensors.')
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(f'{api}: Expected output of f to be a Tensor or Tensors, got {type(out)} as an output')

def assert_non_empty_list_of_tensors(output: List[torch.Tensor], api: str, argname: str) -> None:
    if False:
        while True:
            i = 10
    if len(output) == 0:
        raise RuntimeError(f'{api}: Expected {argname} to contain at least one Tensor.')
    for out in output:
        if isinstance(out, torch.Tensor):
            continue
        raise RuntimeError(f'{api}: Expected {argname} to only contain Tensors, got {type(out)}')
jvp_str = 'jvp(f, primals, tangents)'

def safe_unpack_dual(dual, strict):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(dual, torch.Tensor):
        raise RuntimeError(f'{jvp_str}: expected f(*args) to return only tensors, got unsupported type {type(dual)}')
    (primal, tangent) = fwAD.unpack_dual(dual)
    if tangent is None:
        if strict:
            raise RuntimeError('jvp(f, primals, tangents, strict=True): The output of f is independent of the inputs. This is not allowed with strict=True.')
        tangent = torch.zeros_like(primal)
    return (primal, tangent)

@exposed_in('torch.func')
def jvp(func: Callable, primals: Any, tangents: Any, *, strict: bool=False, has_aux: bool=False):
    if False:
        return 10
    '\n    Standing for the Jacobian-vector product, returns a tuple containing\n    the output of `func(*primals)` and the "Jacobian of ``func`` evaluated at\n    ``primals``" times ``tangents``. This is also known as forward-mode autodiff.\n\n    Args:\n        func (function): A Python function that takes one or more arguments,\n            one of which must be a Tensor, and returns one or more Tensors\n        primals (Tensors): Positional arguments to ``func`` that must all be\n            Tensors. The returned function will also be computing the\n            derivative with respect to these arguments\n        tangents (Tensors): The "vector" for which Jacobian-vector-product is\n            computed. Must be the same structure and sizes as the inputs to\n            ``func``.\n        has_aux (bool): Flag indicating that ``func`` returns a\n            ``(output, aux)`` tuple where the first element is the output of\n            the function to be differentiated and the second element is\n            other auxiliary objects that will not be differentiated.\n            Default: False.\n\n    Returns:\n        Returns a ``(output, jvp_out)`` tuple containing the output of ``func``\n        evaluated at ``primals`` and the Jacobian-vector product.\n        If ``has_aux is True``, then instead returns a ``(output, jvp_out, aux)`` tuple.\n\n    .. note::\n        You may see this API error out with "forward-mode AD not implemented\n        for operator X". If so, please file a bug report and we will prioritize it.\n\n    jvp is useful when you wish to compute gradients of a function R^1 -> R^N\n\n        >>> from torch.func import jvp\n        >>> x = torch.randn([])\n        >>> f = lambda x: x * torch.tensor([1., 2., 3])\n        >>> value, grad = jvp(f, (x,), (torch.tensor(1.),))\n        >>> assert torch.allclose(value, f(x))\n        >>> assert torch.allclose(grad, torch.tensor([1., 2, 3]))\n\n    :func:`jvp` can support functions with multiple inputs by passing in the\n    tangents for each of the inputs\n\n         >>> from torch.func import jvp\n         >>> x = torch.randn(5)\n         >>> y = torch.randn(5)\n         >>> f = lambda x, y: (x * y)\n         >>> _, output = jvp(f, (x, y), (torch.ones(5), torch.ones(5)))\n         >>> assert torch.allclose(output, x + y)\n\n    '
    return _jvp_with_argnums(func, primals, tangents, argnums=None, strict=strict, has_aux=has_aux)

@doesnt_support_saved_tensors_hooks
def _jvp_with_argnums(func: Callable, primals: Any, tangents: Any, argnums: Optional[argnums_t], *, strict: bool=False, has_aux: bool):
    if False:
        while True:
            i = 10
    if not isinstance(primals, tuple):
        raise RuntimeError(f'{jvp_str}: Expected primals to be a tuple. E.g. it should be valid to call f(*primals).')
    diff_args = primals if argnums is None else _slice_argnums(primals, argnums)
    (flat_primals, primals_spec) = tree_flatten(diff_args)
    (flat_tangents, tangents_spec) = tree_flatten(tangents)
    if primals_spec != tangents_spec:
        raise RuntimeError(f'{jvp_str}: Expected primals and tangents to have the same python structure. For example, if primals is a tuple of 3 tensors, tangents also must be. Got primals with structure {primals_spec} and tangents with structure {tangents_spec}')
    assert_non_empty_list_of_tensors(flat_primals, jvp_str, 'primals')
    assert_non_empty_list_of_tensors(flat_tangents, jvp_str, 'tangents')
    level = _jvp_increment_nesting()
    try:
        global JVP_NESTING
        JVP_NESTING += 1
        with fwAD._set_fwd_grad_enabled(True):
            ctx = fwAD.dual_level if JVP_NESTING == 1 else noop
            with ctx():
                flat_duals = tuple((fwAD.make_dual(p, t) for (p, t) in zip(flat_primals, flat_tangents)))
                duals = tree_unflatten(flat_duals, primals_spec)
                if argnums is not None:
                    primals = _wrap_all_tensors(primals, level)
                    duals = _replace_args(primals, duals, argnums)
                result_duals = func(*duals)
                if has_aux:
                    if not (isinstance(result_duals, tuple) and len(result_duals) == 2):
                        raise RuntimeError(f'{jvp_str}: output of function f should be a tuple: (output, aux) if has_aux is True')
                    (result_duals, aux) = result_duals
                    aux = _undo_create_differentiable(aux, level)
                (result_duals, spec) = tree_flatten(result_duals)
                assert_non_empty_tensor_output(result_duals, jvp_str)
                (primals_out, tangents_out) = zip(*[safe_unpack_dual(dual, strict) for dual in result_duals])
                primals_out = tree_map(partial(_undo_create_differentiable, level=level), primals_out)
                tangents_out = tree_map(partial(_undo_create_differentiable, level=level), tangents_out)
                primals_out_unflatten = tree_unflatten(primals_out, spec)
                tangents_out_unflatten = tree_unflatten(tangents_out, spec)
                if has_aux:
                    return (primals_out_unflatten, tangents_out_unflatten, aux)
                return (primals_out_unflatten, tangents_out_unflatten)
    finally:
        _jvp_decrement_nesting()
        JVP_NESTING -= 1

def safe_unflatten(tensor, dim, shape):
    if False:
        i = 10
        return i + 15
    if len(shape) == 0:
        assert tensor.shape[dim] == 1
        return tensor.squeeze(dim)
    return tensor.unflatten(dim, shape)

@exposed_in('torch.func')
def jacfwd(func: Callable, argnums: argnums_t=0, has_aux: bool=False, *, randomness: str='error'):
    if False:
        i = 10
        return i + 15
    '\n    Computes the Jacobian of ``func`` with respect to the arg(s) at index\n    ``argnum`` using forward-mode autodiff\n\n    Args:\n        func (function): A Python function that takes one or more arguments,\n            one of which must be a Tensor, and returns one or more Tensors\n        argnums (int or Tuple[int]): Optional, integer or tuple of integers,\n            saying which arguments to get the Jacobian with respect to.\n            Default: 0.\n        has_aux (bool): Flag indicating that ``func`` returns a\n            ``(output, aux)`` tuple where the first element is the output of\n            the function to be differentiated and the second element is\n            auxiliary objects that will not be differentiated.\n            Default: False.\n        randomness(str): Flag indicating what type of randomness to use.\n            See :func:`vmap` for more detail. Allowed: "different", "same", "error".\n            Default: "error"\n\n    Returns:\n        Returns a function that takes in the same inputs as ``func`` and\n        returns the Jacobian of ``func`` with respect to the arg(s) at\n        ``argnums``. If ``has_aux is True``, then the returned function\n        instead returns a ``(jacobian, aux)`` tuple where ``jacobian``\n        is the Jacobian and ``aux`` is auxiliary objects returned by ``func``.\n\n    .. note::\n        You may see this API error out with "forward-mode AD not implemented\n        for operator X". If so, please file a bug report and we will prioritize it.\n        An alternative is to use :func:`jacrev`, which has better operator coverage.\n\n    A basic usage with a pointwise, unary operation will give a diagonal array\n    as the Jacobian\n\n        >>> from torch.func import jacfwd\n        >>> x = torch.randn(5)\n        >>> jacobian = jacfwd(torch.sin)(x)\n        >>> expected = torch.diag(torch.cos(x))\n        >>> assert torch.allclose(jacobian, expected)\n\n    :func:`jacfwd` can be composed with vmap to produce batched\n    Jacobians:\n\n        >>> from torch.func import jacfwd, vmap\n        >>> x = torch.randn(64, 5)\n        >>> jacobian = vmap(jacfwd(torch.sin))(x)\n        >>> assert jacobian.shape == (64, 5, 5)\n\n    If you would like to compute the output of the function as well as the\n    jacobian of the function, use the ``has_aux`` flag to return the output\n    as an auxiliary object:\n\n        >>> from torch.func import jacfwd\n        >>> x = torch.randn(5)\n        >>>\n        >>> def f(x):\n        >>>   return x.sin()\n        >>>\n        >>> def g(x):\n        >>>   result = f(x)\n        >>>   return result, result\n        >>>\n        >>> jacobian_f, f_x = jacfwd(g, has_aux=True)(x)\n        >>> assert torch.allclose(f_x, f(x))\n\n    Additionally, :func:`jacrev` can be composed with itself or :func:`jacrev`\n    to produce Hessians\n\n        >>> from torch.func import jacfwd, jacrev\n        >>> def f(x):\n        >>>   return x.sin().sum()\n        >>>\n        >>> x = torch.randn(5)\n        >>> hessian = jacfwd(jacrev(f))(x)\n        >>> assert torch.allclose(hessian, torch.diag(-x.sin()))\n\n    By default, :func:`jacfwd` computes the Jacobian with respect to the first\n    input. However, it can compute the Jacboian with respect to a different\n    argument by using ``argnums``:\n\n        >>> from torch.func import jacfwd\n        >>> def f(x, y):\n        >>>   return x + y ** 2\n        >>>\n        >>> x, y = torch.randn(5), torch.randn(5)\n        >>> jacobian = jacfwd(f, argnums=1)(x, y)\n        >>> expected = torch.diag(2 * y)\n        >>> assert torch.allclose(jacobian, expected)\n\n    Additionally, passing a tuple to ``argnums`` will compute the Jacobian\n    with respect to multiple arguments\n\n        >>> from torch.func import jacfwd\n        >>> def f(x, y):\n        >>>   return x + y ** 2\n        >>>\n        >>> x, y = torch.randn(5), torch.randn(5)\n        >>> jacobian = jacfwd(f, argnums=(0, 1))(x, y)\n        >>> expectedX = torch.diag(torch.ones_like(x))\n        >>> expectedY = torch.diag(2 * y)\n        >>> assert torch.allclose(jacobian[0], expectedX)\n        >>> assert torch.allclose(jacobian[1], expectedY)\n\n    '

    @wraps(func)
    def wrapper_fn(*args):
        if False:
            return 10
        error_if_complex('jacfwd', args, is_input=True)
        primals = args if argnums is None else _slice_argnums(args, argnums)
        (flat_primals, primals_spec) = tree_flatten(primals)
        flat_primals_numels = tuple((p.numel() for p in flat_primals))
        flat_basis = _construct_standard_basis_for(flat_primals, flat_primals_numels)
        basis = tree_unflatten(flat_basis, primals_spec)

        def push_jvp(basis):
            if False:
                for i in range(10):
                    print('nop')
            output = _jvp_with_argnums(func, args, basis, argnums=argnums, has_aux=has_aux)
            error_if_complex('jacfwd', output[0], is_input=False)
            if has_aux:
                (_, jvp_out, aux) = output
                return (jvp_out, aux)
            (_, jvp_out) = output
            return jvp_out
        results = vmap(push_jvp, randomness=randomness)(basis)
        if has_aux:
            (results, aux) = results
            (flat_aux, aux_spec) = tree_flatten(aux)
            flat_aux = [value[0] for value in flat_aux]
            aux = tree_unflatten(flat_aux, aux_spec)
        (jac_outs, spec) = tree_flatten(results)
        jac_outs_ins = tuple((tuple((safe_unflatten(jac_out_in, -1, primal.shape) for (primal, jac_out_in) in zip(flat_primals, jac_out.movedim(0, -1).split(flat_primals_numels, dim=-1)))) for jac_out in jac_outs))
        jac_outs_ins = tuple((tree_unflatten(jac_ins, primals_spec) for jac_ins in jac_outs_ins))
        if isinstance(argnums, int):
            jac_outs_ins = tuple((jac_ins[0] for jac_ins in jac_outs_ins))
        if has_aux:
            return (tree_unflatten(jac_outs_ins, spec), aux)
        return tree_unflatten(jac_outs_ins, spec)
    return wrapper_fn

@exposed_in('torch.func')
def hessian(func, argnums=0):
    if False:
        i = 10
        return i + 15
    '\n    Computes the Hessian of ``func`` with respect to the arg(s) at index\n    ``argnum`` via a forward-over-reverse strategy.\n\n    The forward-over-reverse strategy (composing ``jacfwd(jacrev(func))``) is\n    a good default for good performance. It is possible to compute Hessians\n    through other compositions of :func:`jacfwd` and :func:`jacrev` like\n    ``jacfwd(jacfwd(func))`` or ``jacrev(jacrev(func))``.\n\n    Args:\n        func (function): A Python function that takes one or more arguments,\n            one of which must be a Tensor, and returns one or more Tensors\n        argnums (int or Tuple[int]): Optional, integer or tuple of integers,\n            saying which arguments to get the Hessian with respect to.\n            Default: 0.\n\n    Returns:\n        Returns a function that takes in the same inputs as ``func`` and\n        returns the Hessian of ``func`` with respect to the arg(s) at\n        ``argnums``.\n\n    .. note::\n        You may see this API error out with "forward-mode AD not implemented\n        for operator X". If so, please file a bug report and we will prioritize it.\n        An alternative is to use ``jacrev(jacrev(func))``, which has better\n        operator coverage.\n\n    A basic usage with a R^N -> R^1 function gives a N x N Hessian:\n\n        >>> from torch.func import hessian\n        >>> def f(x):\n        >>>   return x.sin().sum()\n        >>>\n        >>> x = torch.randn(5)\n        >>> hess = hessian(f)(x)  # equivalent to jacfwd(jacrev(f))(x)\n        >>> assert torch.allclose(hess, torch.diag(-x.sin()))\n\n    '
    return jacfwd(jacrev(func, argnums), argnums)

@exposed_in('torch.func')
def grad_and_value(func: Callable, argnums: argnums_t=0, has_aux: bool=False) -> Callable:
    if False:
        return 10
    '\n    Returns a function to compute a tuple of the gradient and primal, or\n    forward, computation.\n\n    Args:\n        func (Callable): A Python function that takes one or more arguments.\n            Must return a single-element Tensor. If specified ``has_aux``\n            equals ``True``, function can return a tuple of single-element\n            Tensor and other auxiliary objects: ``(output, aux)``.\n        argnums (int or Tuple[int]): Specifies arguments to compute gradients\n            with respect to. ``argnums`` can be single integer or tuple of\n            integers. Default: 0.\n        has_aux (bool): Flag indicating that ``func`` returns a tensor and\n            other auxiliary objects: ``(output, aux)``. Default: False.\n\n    Returns:\n        Function to compute a tuple of gradients with respect to its inputs\n        and the forward computation. By default, the output of the function is\n        a tuple of the gradient tensor(s) with respect to the first argument\n        and the primal computation. If specified ``has_aux`` equals\n        ``True``, tuple of gradients and tuple of the forward computation with\n        output auxiliary objects is returned. If ``argnums`` is a tuple of\n        integers, a tuple of a tuple of the output gradients with respect to\n        each ``argnums`` value and the forward computation is returned.\n\n    See :func:`grad` for examples\n    '

    @doesnt_support_saved_tensors_hooks
    @wraps(func)
    def wrapper(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        level = _grad_increment_nesting()
        try:
            (output, aux, grad_input) = (None, None, None)
            with torch.enable_grad():
                args = _wrap_all_tensors(args, level)
                kwargs = _wrap_all_tensors(kwargs, level)
                diff_args = _slice_argnums(args, argnums, as_tuple=False)
                tree_map_(partial(_create_differentiable, level=level), diff_args)
                output = func(*args, **kwargs)
                if has_aux:
                    if not (isinstance(output, tuple) and len(output) == 2):
                        raise RuntimeError('grad_and_value(f)(*args): output of function f should be a tuple: (output, aux) if has_aux is True')
                    (output, aux) = output
                if not isinstance(output, torch.Tensor):
                    raise RuntimeError(f'grad_and_value(f)(*args): Expected f(*args) to return a Tensor, got {type(output)}')
                if output.dim() != 0:
                    raise RuntimeError(f'grad_and_value(f)(*args): Expected f(*args) to return a scalar Tensor, got tensor with {output.dim()} dims. Maybe you wanted to use the vjp or jacrev APIs instead?')
                (flat_diff_args, spec) = tree_flatten(diff_args)
                flat_outputs = _as_tuple(output)
                flat_grad_input = _autograd_grad(flat_outputs, flat_diff_args, create_graph=True)
                grad_input = tree_unflatten(flat_grad_input, spec)
                grad_input = _undo_create_differentiable(grad_input, level)
                output = _undo_create_differentiable(output, level)
                if aux is not None:
                    aux = _undo_create_differentiable(aux, level)
            if has_aux:
                return (grad_input, (output, aux))
            return (grad_input, output)
        finally:
            _grad_decrement_nesting()
    return wrapper

def grad_impl(func: Callable, argnums: argnums_t, has_aux: bool, args, kwargs):
    if False:
        for i in range(10):
            print('nop')
    func = lazy_dynamo_disable(func)
    results = grad_and_value(func, argnums, has_aux=has_aux)(*args, **kwargs)
    if has_aux:
        (grad, (_, aux)) = results
        return (grad, aux)
    (grad, _) = results
    return grad

def _maybe_wrap_functional_tensor(maybe_tensor, level, *, _python_functionalize: bool=False):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    wrapped = _wrap_functional_tensor(maybe_tensor, level)
    _assert_wrapped_functional(maybe_tensor, wrapped)
    if _python_functionalize:
        out = FunctionalTensor(wrapped)
        torch._mirror_autograd_meta_to(maybe_tensor, out)
        return out
    return wrapped

def _wrap_all_tensors_to_functional(tensor_pytree, level, *, _python_functionalize: bool=False):
    if False:
        for i in range(10):
            print('nop')
    return tree_map(partial(lambda x: _maybe_wrap_functional_tensor(x, level, _python_functionalize=_python_functionalize)), tensor_pytree)

def _maybe_unwrap_functional_tensor(maybe_tensor, *, reapply_views: bool):
    if False:
        for i in range(10):
            print('nop')
    if not isinstance(maybe_tensor, torch.Tensor):
        return maybe_tensor
    if isinstance(maybe_tensor, FunctionalTensor):
        maybe_tensor = maybe_tensor.elem
    if not torch._is_functional_tensor(maybe_tensor):
        return maybe_tensor
    torch._sync(maybe_tensor)
    return _unwrap_functional_tensor(maybe_tensor, reapply_views)

def _unwrap_all_tensors_from_functional(tensor_pytree, *, reapply_views: bool):
    if False:
        for i in range(10):
            print('nop')
    return tree_map(lambda t: _maybe_unwrap_functional_tensor(t, reapply_views=reapply_views), tensor_pytree)

@exposed_in('torch.func')
def functionalize(func: Callable, *, remove: str='mutations') -> Callable:
    if False:
        while True:
            i = 10
    '\n    functionalize is a transform that can be used to remove (intermediate)\n    mutations and aliasing from a function, while preserving the function\'s\n    semantics.\n\n    ``functionalize(func)`` returns a new function with the same semantics\n    as ``func``, but with all intermediate mutations removed.\n    Every inplace operation performed on an intermediate tensor:\n    ``intermediate.foo_()``\n    gets replaced by its out-of-place equivalent:\n    ``intermediate_updated = intermediate.foo()``.\n\n    functionalize is useful for shipping a pytorch program off to\n    backends or compilers that aren\'t able to easily represent\n    mutations or aliasing operators.\n\n    Args:\n        func (Callable): A Python function that takes one or more arguments.\n        remove (str): An optional string argument, that takes on either\n            the value \'mutations\' or \'mutations_and_views\'.\n            If \'mutations\' is passed in then all mutating operators\n            will be replaced with their non-mutating equivalents.\n            If \'mutations_and_views\' is passed in, then additionally, all aliasing\n            operators will be replaced with their non-aliasing equivalents.\n            Default: \'mutations\'.\n\n    Returns:\n        Returns a new "functionalized" function. It takes the same inputs as\n        ``func``, and has the same behavior, but any mutations\n        (and optionally aliasing) performed on intermediate tensors\n        in the function will be removed.\n\n    functionalize will also remove mutations (and views) that were performed on function inputs.\n    However to preserve semantics, functionalize will "fix up" the mutations after\n    the transform has finished running, by detecting if any tensor inputs "should have"\n    been mutated, and copying the new data back to the inputs if necessary.\n\n\n    Example::\n\n        >>> # xdoctest: +SKIP\n        >>> import torch\n        >>> from torch.fx.experimental.proxy_tensor import make_fx\n        >>> from torch.func import functionalize\n        >>>\n        >>> # A function that uses mutations and views, but only on intermediate tensors.\n        >>> def f(a):\n        ...     b = a + 1\n        ...     c = b.view(-1)\n        ...     c.add_(1)\n        ...     return b\n        ...\n        >>> inpt = torch.randn(2)\n        >>>\n        >>> out1 = f(inpt)\n        >>> out2 = functionalize(f)(inpt)\n        >>>\n        >>> # semantics are the same (outputs are equivalent)\n        >>> print(torch.allclose(out1, out2))\n        True\n        >>>\n        >>> f_traced = make_fx(f)(inpt)\n        >>> f_no_mutations_traced = make_fx(functionalize(f))(inpt)\n        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove=\'mutations_and_views\'))(inpt)\n        >>>\n        >>> print(f_traced.code)\n\n\n\n        def forward(self, a_1):\n            add = torch.ops.aten.add(a_1, 1);  a_1 = None\n            view = torch.ops.aten.view(add, [-1])\n            add_ = torch.ops.aten.add_(view, 1);  view = None\n            return add\n\n        >>> print(f_no_mutations_traced.code)\n\n\n\n        def forward(self, a_1):\n            add = torch.ops.aten.add(a_1, 1);  a_1 = None\n            view = torch.ops.aten.view(add, [-1]);  add = None\n            add_1 = torch.ops.aten.add(view, 1);  view = None\n            view_1 = torch.ops.aten.view(add_1, [2]);  add_1 = None\n            return view_1\n\n        >>> print(f_no_mutations_and_views_traced.code)\n\n\n\n        def forward(self, a_1):\n            add = torch.ops.aten.add(a_1, 1);  a_1 = None\n            view_copy = torch.ops.aten.view_copy(add, [-1]);  add = None\n            add_1 = torch.ops.aten.add(view_copy, 1);  view_copy = None\n            view_copy_1 = torch.ops.aten.view_copy(add_1, [2]);  add_1 = None\n            return view_copy_1\n\n\n        >>> # A function that mutates its input tensor\n        >>> def f(a):\n        ...     b = a.view(-1)\n        ...     b.add_(1)\n        ...     return a\n        ...\n        >>> f_no_mutations_and_views_traced = make_fx(functionalize(f, remove=\'mutations_and_views\'))(inpt)\n        >>> #\n        >>> # All mutations and views have been removed,\n        >>> # but there is an extra copy_ in the graph to correctly apply the mutation to the input\n        >>> # after the function has completed.\n        >>> print(f_no_mutations_and_views_traced.code)\n\n\n\n        def forward(self, a_1):\n            view_copy = torch.ops.aten.view_copy(a_1, [-1])\n            add = torch.ops.aten.add(view_copy, 1);  view_copy = None\n            view_copy_1 = torch.ops.aten.view_copy(add, [2]);  add = None\n            copy_ = torch.ops.aten.copy_(a_1, view_copy_1);  a_1 = None\n            return view_copy_1\n\n\n    There are a few "failure modes" for functionalize that are worth calling out:\n      (1) Like other torch.func transforms, `functionalize()` doesn\'t work with functions\n          that directly use `.backward()`. The same is true for torch.autograd.grad.\n          If you want to use autograd, you can compute gradients directly\n          with `functionalize(grad(f))`.\n      (2) Like other torch.func transforms, `functionalize()` doesn\'t work with global state.\n          If you call `functionalize(f)` on a function that takes views / mutations of\n          non-local state, functionalization will simply no-op and pass the view/mutation\n          calls directly to the backend.\n          One way to work around this is is to ensure that any non-local state creation\n          is wrapped into a larger function, which you then call functionalize on.\n      (3) `resize_()` has some limitations: functionalize will only work on programs\n          that use resize_()` as long as the tensor being resized is not a view.\n      (4) `as_strided()` has some limitations: functionalize will not work on\n          `as_strided()` calls that result in tensors with overlapping memory.\n\n\n    Finally, a helpful mental model for understanding functionalization is that\n    most user pytorch programs are writing with the public torch API.\n    When executed, torch operators are generally decomposed into\n    our internal C++ "ATen" API.\n    The logic for functionalization happens entirely at the level of ATen.\n    Functionalization knows how to take every aliasing operator in ATen,\n    and map it to its non-aliasing equivalent\n    (e.g. ``tensor.view({-1})`` -> ``at::view_copy(tensor, {-1})``),\n    and how to take every mutating operator in ATen,\n    and map it to its non-mutating equivalent\n    (e.g. ``tensor.add_(1)`` -> ``at::add(tensor, -1)``),\n    while tracking aliases and mutations out-of-line to know when to fix things up.\n    Information about which ATen operators are aliasing or mutating all comes from\n    https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/native_functions.yaml.\n    '
    if remove == 'mutations':
        reapply_views = True
    elif remove == 'mutations_and_views':
        reapply_views = False
    else:
        raise RuntimeError(f"functionalize(f, remove='mutations'): received invalid argument for remove={remove}. Valid options are:\n     remove='mutations': all inplace and out= operators will be removed from the program, and replaced with their out-of-place equivalents.\n     remove='mutations_and_views': In addition to the above, all aliasing operators {{view}} will be replaced with their non-aliasing counterparts, {{view}}_copy.\n")

    @doesnt_support_saved_tensors_hooks
    @wraps(func)
    def wrapped(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        try:
            func_level = _func_increment_nesting(reapply_views)
            func_args = _wrap_all_tensors_to_functional(args, func_level)
            func_kwargs = _wrap_all_tensors_to_functional(kwargs, func_level)
            flattened_unwrapped_args = pytree.arg_tree_leaves(*args)
            flattened_wrapped_args = pytree.arg_tree_leaves(*func_args)
            flattened_unwrapped_kwargs = pytree.arg_tree_leaves(**kwargs)
            flattened_wrapped_kwargs = pytree.arg_tree_leaves(**func_kwargs)
            func_outputs = func(*func_args, **func_kwargs)
            outputs = _unwrap_all_tensors_from_functional(func_outputs, reapply_views=reapply_views)
            (flat_outputs, func_out_spec) = tree_flatten(outputs)
            for a in flattened_wrapped_args + flattened_wrapped_kwargs:
                if isinstance(a, torch.Tensor):
                    torch._sync(a)
            for (unwrapped, wrapped) in zip(flattened_unwrapped_args, flattened_wrapped_args):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            for (unwrapped, wrapped) in zip(flattened_unwrapped_kwargs, flattened_wrapped_kwargs):
                if isinstance(unwrapped, torch.Tensor) and isinstance(wrapped, torch.Tensor):
                    _propagate_functional_input_mutation(unwrapped, wrapped)
            return outputs
        finally:
            _func_decrement_nesting()
    return wrapped

@exposed_in('torch.func')
def linearize(func: Callable, *primals) -> Tuple[Any, Callable]:
    if False:
        while True:
            i = 10
    '\n    Returns the value of ``func`` at ``primals`` and linear approximation\n    at ``primals``.\n\n    Args:\n        func (Callable): A Python function that takes one or more arguments.\n        primals (Tensors): Positional arguments to ``func`` that must all be\n            Tensors. These are the values at which the function is linearly approximated.\n\n    Returns:\n        Returns a ``(output, jvp_fn)`` tuple containing the output of ``func``\n        applied to ``primals`` and a function that computes the jvp of\n        ``func`` evaluated at ``primals``.\n\n    linearize is useful if jvp is to be computed multiple times at ``primals``. However,\n    to achieve this, linearize saves intermediate computation and has higher memory requirements\n    than directly applying `jvp`. So, if all the ``tangents`` are known, it maybe more efficient\n    to compute vmap(jvp) instead of using linearize.\n\n    .. note::\n        linearize evaluates ``func`` twice. Please file an issue for an implementation\n        with a single evaluation.\n\n    Example::\n        >>> import torch\n        >>> from torch.func import linearize\n        >>> def fn(x):\n        ...     return x.sin()\n        ...\n        >>> output, jvp_fn = linearize(fn, torch.zeros(3, 3))\n        >>> jvp_fn(torch.ones(3, 3))\n        tensor([[1., 1., 1.],\n                [1., 1., 1.],\n                [1., 1., 1.]])\n        >>>\n\n    '
    output = func(*primals)
    (_, output_spec) = tree_flatten(output)
    (flat_primals, primals_argspec) = tree_flatten(primals)
    flat_tangents = tuple((p.new_empty(()).expand_as(p) for p in flat_primals))

    def trace_fn(flat_tangents):
        if False:
            print('Hello World!')
        with fwAD.dual_level():
            flat_duals = tuple((fwAD.make_dual(p, t) for (p, t) in zip(flat_primals, flat_tangents)))
            duals = tree_unflatten(flat_duals, primals_argspec)
            output = func(*duals)
            tangents = tree_map_only(torch.Tensor, lambda t: fwAD.unpack_dual(t)[1], output)
        return tangents
    jvp_graph = make_fx(trace_fn)(flat_tangents)
    const_folded_jvp_graph = const_fold.split_const_subgraphs(jvp_graph)
    flat_primals_shape = tuple((p.shape for p in flat_primals))
    flat_primals_device = tuple((p.device for p in flat_primals))
    flat_primals_dtype = tuple((p.dtype for p in flat_primals))

    def forward_ad_checks(flat_tangents):
        if False:
            for i in range(10):
                print('nop')
        for (idx, t) in enumerate(flat_tangents):
            if t.shape != flat_primals_shape[idx]:
                msg = f"tangent:{idx} with shape {t.shape} in flattened pytree doesn't match the shape {flat_primals_shape[idx]} of the corresponding primal."
                raise RuntimeError(msg)
            if t.device != flat_primals_device[idx]:
                msg = f"tangent:{idx} with device {t.device} in flattened pytree doesn't match the device {flat_primals_device[idx]} of the corresponding primal."
                raise RuntimeError(msg)
            if t.dtype != flat_primals_dtype[idx]:
                msg = f"tangent:{idx} with dtype {t.dtype} in flattened pytree doesn't match the dtype {flat_primals_dtype[idx]} of the corresponding primal."
                raise RuntimeError(msg)

    def jvp_fn(*tangents):
        if False:
            return 10
        (flat_tangents, tangent_argspec) = tree_flatten(tangents)
        if tangent_argspec != primals_argspec:
            raise RuntimeError(f'Expected the tangents {tangent_argspec} to have the same argspec as the primals {primals_argspec}')
        forward_ad_checks(flat_tangents)
        flat_output = const_folded_jvp_graph(*flat_tangents)
        return tree_unflatten(flat_output, output_spec)
    return (output, jvp_fn)