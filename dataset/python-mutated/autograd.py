import torch
import torch.utils._pytree as pytree
from collections import namedtuple
import functools

def autograd_kernel_indirection(custom_op):
    if False:
        return 10
    autograd_fallback = autograd_not_implemented(custom_op)

    def inner(*args, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        if custom_op._has_impl('autograd'):
            kernel = custom_op._get_impl('autograd').func
            return kernel(*args, **kwargs)
        if custom_op._has_impl('save_for_backward') or custom_op._has_impl('backward'):
            missing = 'save_for_backward' if custom_op._has_impl('backward') else 'backward'
            found = 'save_for_backward' if missing == 'backward' else 'backward'
            loc = custom_op._get_impl(found).location
            raise RuntimeError(f"We found a '{found}' registration for {custom_op} at {loc} but were unable to find a '{missing}' registration. To use the CustomOp API to register a backward formula, please provide us both a backward function and a 'save for backward' function via `impl_backward` and `impl_save_for_backward` respectively.")
        return autograd_fallback(*args, **kwargs)
    return inner

def autograd_not_implemented(custom_op):
    if False:
        for i in range(10):
            print('nop')

    def kernel(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        if torch.is_grad_enabled() and pytree.tree_any(lambda x: isinstance(x, torch.Tensor) and x.requires_grad, (args, kwargs)):
            raise RuntimeError('Autograd has not been implemented for operator')
        with torch._C._AutoDispatchBelowAutograd():
            return custom_op(*args, **kwargs)
    return kernel

def mark_non_differentiable(ctx, output, output_differentiability):
    if False:
        while True:
            i = 10
    if output_differentiability is not None:
        if not isinstance(output, tuple):
            tuple_output = (output,)
        else:
            tuple_output = output
        assert len(output_differentiability) == len(tuple_output)
        non_differentiable_tensors = []
        for (idx, (differentiable, out)) in enumerate(zip(output_differentiability, tuple_output)):
            if isinstance(out, torch.Tensor):
                if not differentiable:
                    non_differentiable_tensors.append(out)
                continue
            if isinstance(out, list):
                if not differentiable:
                    non_differentiable_tensors.extend(out)
                continue
            if differentiable:
                raise RuntimeError(f'With output_differentiability={output_differentiability}. At idx {idx}, we received an object of type {type(out)} that is not a Tensor, so it cannot have be marked as differentiable in output_differentiability.')
        if non_differentiable_tensors:
            ctx.mark_non_differentiable(*non_differentiable_tensors)

def construct_autograd_kernel(schema, output_differentiability, custom_op, op_overload, save_for_backward_fn, backward_fn):
    if False:
        for i in range(10):
            print('nop')

    def apply(*args):
        if False:
            i = 10
            return i + 15
        (flat_args, spec) = pytree.tree_flatten(args)
        out_spec = None

        def forward(ctx, *flat_args):
            if False:
                i = 10
                return i + 15
            ctx.set_materialize_grads(True)
            args = pytree.tree_unflatten(list(flat_args), spec)
            with torch._C._AutoDispatchBelowAutograd():
                output = op_overload(*args)
            args_info = namedtuple_args(schema, pytree.tree_map(type, args))
            save_for_backward_fn_inputs = namedtuple_args(schema, args)
            to_save = save_for_backward_fn(save_for_backward_fn_inputs, output)
            save_pytree_for_backward(ctx, (to_save, args_info))
            mark_non_differentiable(ctx, output, output_differentiability)
            nonlocal out_spec
            (flat_output, out_spec) = pytree.tree_flatten(output)
            return tuple(flat_output)

        def backward(ctx, *flat_grad_output):
            if False:
                return 10
            assert out_spec is not None
            grads = pytree.tree_unflatten(list(flat_grad_output), out_spec)
            (saved, args_info) = unpack_saved(ctx)
            inner_ctx = object()
            if not isinstance(grads, tuple):
                grads = (grads,)
            grad_inputs_dict = backward_fn(inner_ctx, saved, *grads)
            validate_grad_inputs_dict(grad_inputs_dict, custom_op, args_info)
            return grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info)
        generated_cls = gen_autograd_function(custom_op._opname + '_customop', forward, backward)
        flat_output = generated_cls.apply(*flat_args)
        assert out_spec is not None
        return pytree.tree_unflatten(list(flat_output), out_spec)
    return apply

def gen_autograd_function(name, forward, backward):
    if False:
        print('Hello World!')
    generated_cls = type(name, (torch.autograd.Function,), {'forward': staticmethod(forward), 'backward': staticmethod(backward)})
    return generated_cls

@functools.lru_cache
def namedtuple_args_cls(schema):
    if False:
        for i in range(10):
            print('nop')
    attribs = [arg.name for arg in schema.arguments.flat_all]
    name = str(schema.name) + '_args'
    tuple_cls = namedtuple(name, attribs)
    return tuple_cls

def namedtuple_args(schema, args):
    if False:
        print('Hello World!')
    assert isinstance(args, tuple)
    tuple_cls = namedtuple_args_cls(schema)
    return tuple_cls(*args)

def validate_grad_inputs_dict(grad_inputs_dict, forward_op, args_info):
    if False:
        i = 10
        return i + 15

    def error(what):
        if False:
            i = 10
            return i + 15
        backward = forward_op._get_impl('backward')
        raise RuntimeError(f'In the backward function defined for {forward_op} at {backward.location} using the CustomOp API, {what}')
    if not isinstance(grad_inputs_dict, dict):
        error(f'expected the output of the backward function to be a dict but got {type(grad_inputs_dict)}')
    expected_keys = {arg.name for arg in forward_op._schema.arguments.flat_all if arg.type.is_tensor_like()}
    actual_keys = grad_inputs_dict.keys()
    if expected_keys != actual_keys:
        error(f'expected the returned grad_input dict to have keys {expected_keys} but got {actual_keys}. The backward function must return a gradient (can be None) for each arg to the CustomOp that may be a Tensor or Sequence[Tensor]. Args declared to be non-Tensor-like types should not appear in the grad_input dict')
    for (name, grad) in grad_inputs_dict.items():
        arg_info = getattr(args_info, name)
        if isinstance(arg_info, list):
            if not isinstance(grad, (tuple, list)):
                error(f"for input '{name}' expected the grad_input dict to hold a list of gradients but got object of type {type(grad)}.")
            if not len(grad) == len(arg_info):
                error(f"for input '{name}' expected the grad_input dict to hold a list of {len(arg_info)} gradients but got {len(grad)}")
            for (idx, (g, info)) in enumerate(zip(grad, arg_info)):
                if g is None:
                    continue
                if not isinstance(g, torch.Tensor):
                    error(f"for input '{name}' expected the grad_input dict to hold a list of None or Tensor gradients but got object of {type(g)} at index {idx}")
                if not issubclass(info, torch.Tensor):
                    error(f"for input '{name}', got a Tensor as the gradient for the {idx}-th value but expected None because the {idx}-th value was not a Tensor (it was type {arg_info}")
            continue
        if grad is None:
            continue
        if not isinstance(grad, torch.Tensor):
            error(f"got object of type {type(grad)} as the gradient for input '{name}', but expected the gradient to be either None or a Tensor")
        if not issubclass(arg_info, torch.Tensor):
            error(f"got a Tensor as the gradient for input '{name}' but expected None as the gradient because input '{name}' was not a Tensor (it was type {arg_info}).")

def grad_inputs_dict_to_flat_tuple(grad_inputs_dict, args_info):
    if False:
        print('Hello World!')
    result = []
    for (name, arg_info) in args_info._asdict().items():
        if name not in grad_inputs_dict:
            result.append(pytree.tree_map(lambda x: None, arg_info))
            continue
        result.append(grad_inputs_dict[name])
    return tuple(pytree.tree_leaves(result))

def save_pytree_for_backward(ctx, stuff):
    if False:
        for i in range(10):
            print('nop')
    (flat_stuff, spec) = pytree.tree_flatten(stuff)
    num_elts = len(flat_stuff)
    tensor_idxs = [idx for (idx, thing) in enumerate(flat_stuff) if isinstance(thing, torch.Tensor)]
    non_tensor_idxs = [idx for (idx, thing) in enumerate(flat_stuff) if not isinstance(thing, torch.Tensor)]
    tensors = [thing for thing in flat_stuff if isinstance(thing, torch.Tensor)]
    non_tensors = [thing for thing in flat_stuff if not isinstance(thing, torch.Tensor)]
    ctx.spec = spec
    ctx.num_elts = num_elts
    ctx.save_for_backward(*tensors)
    ctx.tensor_idxs = tensor_idxs
    ctx.saved_non_tensors = non_tensors
    ctx.non_tensor_idxs = non_tensor_idxs

def unpack_saved(ctx):
    if False:
        for i in range(10):
            print('nop')
    flat_stuff = [None] * ctx.num_elts
    for (tensor, idx) in zip(ctx.saved_tensors, ctx.tensor_idxs):
        flat_stuff[idx] = tensor
    for (non_tensor, idx) in zip(ctx.saved_non_tensors, ctx.non_tensor_idxs):
        flat_stuff[idx] = non_tensor
    stuff = pytree.tree_unflatten(flat_stuff, ctx.spec)
    return stuff