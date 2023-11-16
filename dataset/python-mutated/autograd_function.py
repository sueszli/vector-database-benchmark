import torch
from torch._ops import HigherOrderOperator
from torch._C._functorch import TransformType
from torch._functorch.utils import enable_single_level_autograd_function
import torch.utils._pytree as pytree
from torch._C._functorch import _wrap_for_grad, _unwrap_for_grad, current_level
from torch._functorch.vmap import wrap_batched, unwrap_batched, restore_vmap, _add_batch_dim
from torch._functorch.apis import vmap
from torch._functorch.vmap import _broadcast_to_and_flatten
from torch.autograd.forward_ad import _set_fwd_grad_enabled
from typing import Any, NamedTuple, Tuple

class CustomFunctionHigherOrderOperator(HigherOrderOperator):

    def __init__(self):
        if False:
            while True:
                i = 10
        super().__init__('custom_function_call')

    def __call__(self, autograd_function, *args, **kwargs):
        if False:
            return 10
        if torch._C._are_functorch_transforms_active():
            return super().__call__(autograd_function, *args, **kwargs)
        return autograd_function.apply(*args, **kwargs)
custom_function_call = CustomFunctionHigherOrderOperator()

@custom_function_call.py_impl(TransformType.Grad)
@custom_function_call.py_impl(TransformType.Jvp)
def custom_function_call_grad(interpreter, autograd_function, *operands):
    if False:
        i = 10
        return i + 15
    Generated = generate_single_level_function(interpreter, autograd_function)
    with enable_single_level_autograd_function():
        flat_out = Generated.apply(*operands)
    return flat_out

def generate_single_level_function(interpreter, autograd_function):
    if False:
        i = 10
        return i + 15
    level = interpreter.level()

    def forward(*operands):
        if False:
            i = 10
            return i + 15
        unwrapped_operands = pytree.tree_map_only(torch.Tensor, lambda x: _unwrap_for_grad(x, level), operands)
        with torch.enable_grad(), _set_fwd_grad_enabled(True), interpreter.lower():
            unwrapped_output = custom_function_call(autograd_function, *unwrapped_operands)

        def wrap_fn(output):
            if False:
                for i in range(10):
                    print('nop')
            return _wrap_for_grad(output, level)
        return wrap_outputs_maintaining_identity(unwrapped_output, unwrapped_operands, operands, wrap_fn)

    def setup_context(ctx, inputs, output):
        if False:
            print('Hello World!')
        return autograd_function.setup_context(ctx, inputs, output)

    def backward(ctx, *grads):
        if False:
            print('Hello World!')
        result = autograd_function.backward(ctx, *grads)
        return result

    def jvp(ctx, *tangents):
        if False:
            for i in range(10):
                print('nop')
        result = autograd_function.jvp(ctx, *tangents)
        return result
    name = f'{autograd_function.__name__}Generated'
    Generated = type(name, (torch.autograd.function._SingleLevelFunction,), {'forward': staticmethod(forward), 'backward': staticmethod(backward), 'jvp': staticmethod(jvp), 'setup_context': staticmethod(setup_context)})
    return Generated
NO_OUT_DIMS = 'not specified'

def wrap_outputs_maintaining_identity(outputs, unwrapped_inputs, orig_inputs, wrap_fn, out_dims=NO_OUT_DIMS):
    if False:
        i = 10
        return i + 15
    flat_unwrapped_inputs = pytree.arg_tree_leaves(*unwrapped_inputs)
    flat_orig_inputs = pytree.arg_tree_leaves(*orig_inputs)
    unwrapped_input_to_orig_input = {id(unwrapped): orig for (unwrapped, orig) in zip(flat_unwrapped_inputs, flat_orig_inputs)}
    (flat_outputs, spec) = pytree.tree_flatten(outputs)
    result = []
    out_dims_specified = out_dims != NO_OUT_DIMS
    if out_dims_specified:
        flat_out_dims = _broadcast_to_and_flatten(out_dims, spec)
        if flat_out_dims is None:
            raise RuntimeError(f"The autograd.Function's vmap staticmethod returned an incompatible (output, out_dims) tuple. Expected out_dims={out_dims} to be compatible with the structure of `output`. out_dims has structure {pytree.tree_flatten(out_dims)[1]} but output has structure {spec}. For more details, please see https://pytorch.org/docs/master/notes/extending.func.html")
    for (i, output) in enumerate(flat_outputs):
        if not isinstance(output, torch.Tensor):
            result.append(output)
            continue
        if id(output) in unwrapped_input_to_orig_input:
            result.append(unwrapped_input_to_orig_input[id(output)])
            continue
        if out_dims_specified:
            result.append(wrap_fn(output, flat_out_dims[i]))
        else:
            result.append(wrap_fn(output))
    return pytree.tree_unflatten(result, spec)

class VmapInfo(NamedTuple):
    batch_size: int
    randomness: str

def has_overriden_vmap_rule(autograd_function):
    if False:
        return 10
    return autograd_function.vmap is not torch.autograd.Function.vmap

def validate_vmap_returns_tuple_of_two_elements(result):
    if False:
        i = 10
        return i + 15
    base_error_msg = 'Expected the vmap staticmethod to have two returns, an output and out_dims with pytree structure compatible with the output. '
    if not isinstance(result, tuple):
        raise RuntimeError(base_error_msg + f'Got a {type(result)} instead')
    if not len(result) == 2:
        raise RuntimeError(base_error_msg + f'Got {len(result)} returns instead')

@custom_function_call.py_impl(TransformType.Vmap)
def custom_function_call_vmap(interpreter, autograd_function, *operands):
    if False:
        return 10
    if autograd_function.generate_vmap_rule:
        if has_overriden_vmap_rule(autograd_function):
            raise RuntimeError(f'You tried to vmap over {autograd_function.__name__}, but it has both generate_vmap_rule=True and an overriden vmap staticmethod. Please set generate_vmap_rule=False or delete the overriden vmap staticmethod to avoid ambiguity. For more details, please see https://pytorch.org/docs/master/notes/extending.func.html')
        return custom_function_call_vmap_generate_rule(interpreter, autograd_function, *operands)
    if not has_overriden_vmap_rule(autograd_function):
        raise RuntimeError(f'You tried to vmap over {autograd_function.__name__}, but it does not have vmap support. Please override and implement the vmap staticmethod or set generate_vmap_rule=True. For more details, please see https://pytorch.org/docs/master/notes/extending.func.html')
    current_level = interpreter.level()
    info = VmapInfo(batch_size=interpreter.batch_size(), randomness=interpreter.randomness())
    (unwrapped_operands, in_dims) = unwrap_batched(operands, current_level)
    if pytree.tree_all(lambda dim: dim is None, in_dims):
        with interpreter.lower():
            return custom_function_call(autograd_function, *operands)
    with interpreter.lower():
        result = autograd_function.vmap(info, in_dims, *unwrapped_operands)
    validate_vmap_returns_tuple_of_two_elements(result)
    (unwrapped_output, out_dims) = result

    def wrap_fn(output, out_dim):
        if False:
            i = 10
            return i + 15
        return output if out_dim is None else _add_batch_dim(output, out_dim, current_level)
    return wrap_outputs_maintaining_identity(unwrapped_output, unwrapped_operands, operands, wrap_fn, out_dims=out_dims)

def custom_function_call_vmap_generate_rule(interpreter, autograd_function, *operands):
    if False:
        while True:
            i = 10
    (unwrapped_operands, in_dims) = unwrap_batched(operands, interpreter.level())
    (vmapped_function, get_out_dims) = vmapify_autograd_function(autograd_function, in_dims, interpreter.batch_size(), interpreter.randomness())
    with interpreter.lower():
        output = custom_function_call(vmapped_function, *unwrapped_operands)
    out_dims = get_out_dims()
    return wrap_batched(output, out_dims, interpreter.level())

@custom_function_call.py_impl(TransformType.Functionalize)
def custom_function_call_functionalize(interpreter, autograd_function, generate_vmap_rule, *operands):
    if False:
        print('Hello World!')
    raise RuntimeError('NYI: Functionalize rule for custom_function_call')

def vmapify_autograd_function(autograd_function, in_dims, batch_size, randomness):
    if False:
        while True:
            i = 10
    init_val = 'not populated'
    out_dims = init_val
    input_shapes: Any = init_val
    saved_tensors_bdims: Any = init_val

    def forward(*operands):
        if False:
            print('Hello World!')
        nonlocal out_dims
        (outputs, out_dims) = restore_vmap(autograd_function.forward, in_dims, batch_size, randomness)(*operands)
        return outputs

    def setup_context(ctx, inputs, outputs):
        if False:
            i = 10
            return i + 15
        input_shapes_ = None
        saved_tensors_bdims_ = None

        def inner(inputs, outputs):
            if False:
                while True:
                    i = 10
            wrapped_ctx = CtxCustomSave(ctx, current_level())
            autograd_function.setup_context(wrapped_ctx, inputs, outputs)
            nonlocal input_shapes_
            input_shapes_ = tuple((inp.shape if isinstance(inp, torch.Tensor) else None for inp in inputs))
            nonlocal saved_tensors_bdims_
            saved_tensors_bdims_ = wrapped_ctx._pt_saved_tensors_bdims
        restore_vmap(inner, (in_dims, out_dims), batch_size, randomness)(inputs, outputs)
        nonlocal input_shapes
        input_shapes = input_shapes_
        nonlocal saved_tensors_bdims
        saved_tensors_bdims = saved_tensors_bdims_

    def jvp(ctx, *tangents):
        if False:
            return 10
        assert out_dims != init_val
        assert saved_tensors_bdims != init_val

        def jvp_no_context(saved_tensors, tangents):
            if False:
                for i in range(10):
                    print('nop')
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.jvp(wrapped_ctx, *tangents)
        tangent_in_dims = get_tangents_in_dims(in_dims, tangents)
        (out_tangents, out_tangents_dims) = restore_vmap(jvp_no_context, (saved_tensors_bdims, tangent_in_dims), batch_size, randomness)(ctx.saved_tensors, tangents)
        result = reductify(out_tangents, out_tangents_dims, out_dims, batch_size)
        return result

    def backward(ctx, *grad_outputs):
        if False:
            return 10
        assert out_dims != init_val
        assert input_shapes != init_val
        assert saved_tensors_bdims != init_val

        def backward_no_context(inputs):
            if False:
                while True:
                    i = 10
            (saved_tensors, grad_outputs) = inputs
            wrapped_ctx = CtxWithSavedTensors(ctx, saved_tensors)
            return autograd_function.backward(wrapped_ctx, *grad_outputs)
        (grad_ins, grad_ins_dims) = restore_vmap(backward_no_context, ((saved_tensors_bdims, out_dims),), batch_size, randomness)((ctx.saved_tensors, grad_outputs))
        result = reductify(grad_ins, grad_ins_dims, in_dims, batch_size, input_shapes)
        return result
    name = f'Vmapped{autograd_function.__name__}'
    Generated = type(name, (torch.autograd.Function,), {'forward': staticmethod(forward), 'backward': staticmethod(backward), 'jvp': staticmethod(jvp), 'setup_context': staticmethod(setup_context), 'generate_vmap_rule': True})

    def get_out_dims():
        if False:
            i = 10
            return i + 15
        assert out_dims != init_val
        return out_dims
    return (Generated, get_out_dims)

def get_tangents_in_dims(input_dims, tangents):
    if False:
        print('Hello World!')
    (flat_in_dims, spec) = pytree.tree_flatten(input_dims)
    flat_tangents = pytree.arg_tree_leaves(*tangents)
    result = [None if tangent is None else in_dim for (in_dim, tangent) in zip(flat_in_dims, flat_tangents)]
    return pytree.tree_unflatten(result, spec)

class WrappedCtx:
    _pt_reserved_attrs: Tuple[str, ...] = ('_pt_reserved_attrs', '_pt_inner_ctx')

    def __init__(self, ctx):
        if False:
            return 10
        if not isinstance(ctx, WrappedCtx):
            reserved_attrs = type(self)._pt_reserved_attrs
            for name in reserved_attrs:
                if not hasattr(ctx, name):
                    continue
                raise RuntimeError(f'PyTorch reserves the {reserved_attrs} field on ctx. Please name your fields on ctx something else to avoid name collision.')
        self._pt_inner_ctx = ctx

    def __getattr__(self, name):
        if False:
            return 10
        return getattr(self._pt_inner_ctx, name)

    def __setattr__(self, name, value):
        if False:
            print('Hello World!')
        if name in type(self)._pt_reserved_attrs:
            self.__dict__[name] = value
            return
        return setattr(self._pt_inner_ctx, name, value)

class CtxWithSavedTensors(WrappedCtx):
    _pt_reserved_attrs = ('_pt_new_saved_tensors', *WrappedCtx._pt_reserved_attrs)

    def __init__(self, ctx, new_saved_tensors):
        if False:
            while True:
                i = 10
        super().__init__(ctx)
        self._pt_new_saved_tensors = new_saved_tensors

    @property
    def saved_tensors(self):
        if False:
            print('Hello World!')
        return self._pt_new_saved_tensors

class CtxCustomSave(WrappedCtx):
    _pt_reserved_attrs = ('_pt_saved_tensors_bdims', '_pt_current_level', *WrappedCtx._pt_reserved_attrs)

    def __init__(self, ctx, current_level):
        if False:
            print('Hello World!')
        super().__init__(ctx)
        self._pt_saved_tensors_bdims = ()
        self._pt_current_level = current_level

    def save_for_backward(self, *tensors):
        if False:
            i = 10
            return i + 15
        (unwrapped_tensors, bdims) = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_backward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims

    def save_for_forward(self, *tensors):
        if False:
            while True:
                i = 10
        (unwrapped_tensors, bdims) = unwrap_batched(tensors, self._pt_current_level)
        self._pt_inner_ctx.save_for_forward(*unwrapped_tensors)
        self._pt_saved_tensors_bdims = bdims

def reductify(grad_input, grad_input_bdim, input_bdim, batch_size, target_shape_without_bdim_to_reduce_to=None):
    if False:
        i = 10
        return i + 15
    if not isinstance(grad_input, tuple):
        grad_input = (grad_input,)
    if not isinstance(grad_input_bdim, tuple):
        grad_input_bdim = (grad_input_bdim,)
    if not isinstance(input_bdim, tuple):
        input_bdim = (input_bdim,)
    if target_shape_without_bdim_to_reduce_to is None:
        target_shape_without_bdim_to_reduce_to = len(grad_input) * (None,)
    result = tuple((reductify_leaf(gi, gi_bdim, i_bdim, batch_size, maybe_ishape) for (gi, gi_bdim, i_bdim, maybe_ishape) in zip(grad_input, grad_input_bdim, input_bdim, target_shape_without_bdim_to_reduce_to)))
    return result

def reductify_leaf(grad_input, grad_input_bdim, input_bdim, batch_size, target_shape_without_bdim_to_reduce_to=None):
    if False:
        return 10
    if grad_input is None:
        return None
    if grad_input_bdim is None and input_bdim is None:
        return grad_input
    if grad_input_bdim is not None and input_bdim is None:
        return grad_input.sum(grad_input_bdim)
    assert input_bdim is not None
    if grad_input_bdim is None:
        grad_input = grad_input.unsqueeze(input_bdim)
        new_shape = list(grad_input.shape)
        new_shape[input_bdim] = batch_size
        grad_input = grad_input.expand(new_shape)
        grad_input_bdim = input_bdim
    if target_shape_without_bdim_to_reduce_to is not None:
        return vmap(torch.Tensor.sum_to_size, in_dims=(grad_input_bdim, None), out_dims=input_bdim)(grad_input, target_shape_without_bdim_to_reduce_to)
    if input_bdim != grad_input_bdim:
        grad_input = grad_input.movedim(grad_input_bdim, input_bdim)
    return grad_input