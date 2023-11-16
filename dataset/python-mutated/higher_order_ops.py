import contextlib
import functools
import itertools
import logging
from typing import Dict, List, Optional
import torch._C
import torch.fx
import torch.nn
import torch.onnx.operators
from torch._dispatch.python import enable_python_dispatcher
from torch._dynamo.utils import deepcopy_to_fake_tensor, get_fake_value, get_real_value
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.builtin import BuiltinVariable
from torch._dynamo.variables.functions import UserFunctionVariable
from torch._dynamo.variables.tensor import SymNodeVariable
from torch._guards import Source
from torch.utils import _pytree as pytree
from ..exc import UncapturedHigherOrderOpError, unimplemented, Unsupported, UserError, UserErrorType
from ..source import FSDPNNModuleSource, GetItemSource, NNModuleSource
from ..utils import proxy_args_kwargs
from .dicts import ConstDictVariable
from .lists import ListVariable, TupleVariable
from .nn_module import NNModuleVariable, UnspecializedNNModuleVariable
log = logging.getLogger(__name__)

def safe_or_raise_always_restore(tx, graph_checkpoint, checkpoint, f, sub_args):
    if False:
        while True:
            i = 10
    try:
        f.call_function(tx, sub_args, {})
    finally:
        tx.output.graph = graph_checkpoint
        tx.restore_graphstate(checkpoint)

def raise_hard_error_if_graph_break(reason):
    if False:
        for i in range(10):
            print('nop')

    def deco(fn):
        if False:
            while True:
                i = 10

        @functools.wraps(fn)
        def graph_break_as_hard_error(*args, **kwargs):
            if False:
                print('Hello World!')
            try:
                return fn(*args, **kwargs)
            except Unsupported as e:
                msg = ' Scroll up to find out what causes the graph break.'
                raise UncapturedHigherOrderOpError(reason + msg) from e
        return graph_break_as_hard_error
    return deco

@contextlib.contextmanager
def dynamo_enable_grad(tx):
    if False:
        return 10
    from . import GradModeVariable
    org_value = torch.is_grad_enabled()
    try:
        GradModeVariable.create(tx, True, initialized=True)
        yield
    finally:
        GradModeVariable.create(tx, org_value, initialized=True)

def only_consist_of(var, types):
    if False:
        i = 10
        return i + 15
    if isinstance(var, types):
        return True
    if isinstance(var, (TupleVariable, ListVariable)):
        return all((only_consist_of(item, types) for item in var.items))
    if isinstance(var, ConstDictVariable):
        return all((only_consist_of(item, types) for item in var.items.values()))
    return False

def validate_args_and_maybe_create_graph_inputs(sub_args, tracer, tx, manually_set_subgraph_inputs):
    if False:
        i = 10
        return i + 15
    from . import AutogradFunctionContextVariable, ConstantVariable, SymNodeVariable, TensorVariable
    from .builder import wrap_fx_proxy, wrap_fx_proxy_cls
    assert tracer.parent is not None
    args = []
    for a in sub_args:
        assert isinstance(a, VariableTracker)
        if isinstance(a, ConstantVariable):
            if manually_set_subgraph_inputs:
                tracer.create_graph_input('const')
            new_arg = a
        elif isinstance(a, TensorVariable):
            if manually_set_subgraph_inputs:
                new_proxy = tracer.create_graph_input(a.as_proxy().node.name)
                example_value = a.as_proxy().node.meta['example_value']
                new_arg = wrap_fx_proxy(tx=tx, proxy=new_proxy, example_value=example_value)
            else:
                new_arg = a
        elif isinstance(a, SymNodeVariable):
            if manually_set_subgraph_inputs:
                new_proxy = tracer.create_graph_input(str(a.sym_num.node.expr))
                new_arg = wrap_fx_proxy_cls(target_cls=SymNodeVariable, tx=tx, proxy=new_proxy, example_value=a.sym_num)
            else:
                new_arg = a
        elif isinstance(a, AutogradFunctionContextVariable):
            if manually_set_subgraph_inputs:
                tracer.create_graph_input(a.as_proxy().node.name)
            new_arg = a
        elif manually_set_subgraph_inputs:
            raise unimplemented(f'HigherOrderOperator with body that accepts non-Tensors as input. Got: {a.python_type()}')
        elif only_consist_of(a, (ConstantVariable, SymNodeVariable, TensorVariable)):
            new_arg = a
        else:
            unimplemented("HigherOrderOperator with body that accepts non-Tensors as input that can't be lifted by tracer.")
        args.append(new_arg)
    return args

def speculate_subgraph(tx, f, sub_args, sub_kwargs, graph_checkpoint, checkpoint, description, *, source_target=None, always_restore=False, enable_grad=False, manually_set_subgraph_inputs=True, restore_side_effects=True, should_flatten_outputs=False, tracer=None):
    if False:
        i = 10
        return i + 15
    if sub_kwargs is None:
        sub_kwargs = {}
    if sub_kwargs and manually_set_subgraph_inputs:
        unimplemented('Use `manually_set_subgraph_inputs=False` when passing `sub_kwargs`.')
    try:
        (f, sub_args, sub_kwargs) = VariableTracker.apply(lambda x: x.realize(), (f, sub_args, sub_kwargs))
        with tx.output.subtracer(source_target, tracer) as subtracer:
            args = validate_args_and_maybe_create_graph_inputs(sub_args, subtracer, tx, manually_set_subgraph_inputs)
            validate_args_and_maybe_create_graph_inputs(sub_kwargs.values(), subtracer, tx, manually_set_subgraph_inputs=False)
            autograd_ctx = dynamo_enable_grad(tx) if enable_grad else contextlib.nullcontext()
            if restore_side_effects:
                prev_side_effects = tx.output.side_effects.clone()
            with autograd_ctx:
                output = f.call_function(tx, args, sub_kwargs)
            if restore_side_effects:
                tx.output.side_effects = prev_side_effects
            treespec = None
            if should_flatten_outputs:
                tree_flatten = UserFunctionVariable(pytree.tree_flatten)
                tree_flatten_output = tree_flatten.call_function(tx, [output], {})
                (output, treespec) = tree_flatten_output.unpack_var_sequence(tx)
                output = BuiltinVariable(tuple).call_function(tx, [output], {})
            if always_restore:
                return ((output, treespec), tx.output.graph, subtracer.lifted_freevars)
            else:
                from . import TensorVariable
                if not only_consist_of(output, TensorVariable):
                    unimplemented("HigherOrderOperator body's output must consist of tensors only")
                output_proxies = output.as_proxy()
                output_proxies = pytree.tree_map(subtracer.maybe_lift_tracked_freevar_to_input, output_proxies)
                tx.output.create_node('output', 'output', subtracer.create_arg((output_proxies,)), {})
                graph = tx.output.graph
                graph.lint()
                lifted_freevars = subtracer.lifted_freevars
                return ((output, treespec), graph, lifted_freevars)
    except Unsupported as ex:
        f_name = f'{type(f).__name__}'
        if isinstance(f, UserFunctionVariable):
            f_name = f.get_name()
        msg = f'speculate_subgraph: while introspecting {description}, we were unable to trace function `{f_name}` into a single graph. This means that Dynamo was unable to prove safety for this API and will fall back to eager-mode PyTorch, which could lead to a slowdown.'
        log.warning(msg)
        log.exception(ex)
        tx.output.graph = graph_checkpoint
        tx.restore_graphstate(checkpoint)
        raise Unsupported(f'{msg} Scroll up for the stack trace of the initial exception. The reason was: {ex.msg}') from ex

def make_attr(tx, name):
    if False:
        for i in range(10):
            print('nop')
    node = tx.output.create_proxy('get_attr', name, (), {})
    return node

def add_subgraph(tx, source, name, gm):
    if False:
        for i in range(10):
            print('nop')
    next_name = None
    i = 0
    while not next_name:
        candidate = f'{name}_{i}'
        if candidate in tx.output.nn_modules:
            i += 1
        else:
            next_name = candidate
    gm.__name__ = next_name
    if source.guard_source().is_fsdp_module():
        src = FSDPNNModuleSource(GetItemSource(source, next_name))
    else:
        src = NNModuleSource(GetItemSource(source, next_name))
    gm.torchdynamo_force_dynamic = False
    tx.output.register_attr_or_module(gm, next_name, source=src)
    return next_name

class TorchHigherOrderOperatorVariable(VariableTracker):

    def __init__(self, value, source: Optional[Source]=None, **kwargs):
        if False:
            return 10
        super().__init__(**kwargs)
        self.value = value
        self.source = source

    @staticmethod
    def make(value, source=None, **kwargs):
        if False:
            return 10
        if value.__name__ == 'cond':
            return CondHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ('map', 'map_impl'):
            return MapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == 'executorch_call_delegate':
            return ExecutorchCallDelegateHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == 'out_dtype':
            return OutDtypeHigherOrderVariable(value, source, **kwargs)
        elif value is torch._functorch.eager_transforms.grad_impl:
            return FunctorchGradHigherOrderVariable(value, source, **kwargs)
        elif value is torch._functorch.vmap.vmap_impl:
            return FunctorchVmapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ('trampoline_autograd_fwd', 'trampoline_autograd_bwd', 'trampoline_autograd_apply'):
            return AutogradFunctionMethodHigherOrderVariable(value=value, source=source, **kwargs)
        elif value.__name__ == 'wrap':
            return WrapHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ in ('wrap_activation_checkpoint', 'tag_activation_checkpoint'):
            return CheckpointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == '_export_tracepoint':
            return ExportTracepointHigherOrderVariable(value, source, **kwargs)
        elif value.__name__ == 'trace_wrapped':
            return TraceWrappedHigherOrderOperatorVariable(value, source, **kwargs)
        else:
            unimplemented(f'HigherOrderOperator {value.__name__}')

    def call_function(self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> VariableTracker:
        if False:
            return 10
        unimplemented(f'HigherOrderOperator {self.value.__name__}')

class CondHigherOrderVariable(TorchHigherOrderOperatorVariable):

    @raise_hard_error_if_graph_break(reason="Cond doesn't work unless it is captured completely with torch.compile.")
    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            return 10
        from . import ConstantVariable, ListVariable, NestedUserFunctionVariable, TensorVariable, UserFunctionVariable
        from .builder import wrap_fx_proxy
        (args, kwargs) = VariableTracker.apply(lambda x: x.realize(), (args, kwargs))
        for (i, k) in enumerate(['pred', 'true_fn', 'false_fn', 'operands']):
            if (v := kwargs.pop(k, None)):
                assert i == len(args), 'did not provide the right number of non-keyword args'
                args.append(v)
        if kwargs:
            unimplemented(f'torch.cond: Got unexpected kwargs: {list(kwargs.keys())}')
        if len(args) != 4:
            unimplemented(f'Expected 4 arguments but got {len(args)}.\nUsage: cond(pred, true_fn, false_fn, operands)')
        if type(args[0]) not in (ConstantVariable, TensorVariable, SymNodeVariable):
            unimplemented(f'Expected pred to be bool or a boolean tensor with single item but got {str(type(args[0]))} with original python type {str(args[0].python_type())}.')
        if not isinstance(args[3], (ListVariable, TupleVariable)):
            unimplemented(f'Expected a tuple but got {args[3].python_type()}')
        operands = args[3].unpack_var_sequence(tx)
        if not only_consist_of(args[3], (TensorVariable,)):
            unimplemented('Expect operands to be a tuple of pytrees that only consists of tensor leaves.')
        assert isinstance(args[1], (UserFunctionVariable, NestedUserFunctionVariable, NNModuleVariable, UnspecializedNNModuleVariable)), str(type(args[1]))
        assert isinstance(args[2], (UserFunctionVariable, NestedUserFunctionVariable, NNModuleVariable, UnspecializedNNModuleVariable)), str(type(args[2]))
        (graph_checkpoint, checkpoint) = (tx.output.graph, tx.copy_graphstate())

        def speculate_branch(branch):
            if False:
                while True:
                    i = 10
            ix = 1 if branch else 2
            ((ret_val, _), ret_graph, ret_lifted_freevars) = speculate_subgraph(tx, args[ix], operands, {}, graph_checkpoint, checkpoint, 'cond', source_target=self.value, manually_set_subgraph_inputs=False)
            if not isinstance(ret_val, TensorVariable):
                unimplemented('Expected branch to return a single tensor')
            return (ret_val, ret_graph, ret_lifted_freevars)
        (true_r, true_graph, true_lifted_freevars) = speculate_branch(True)
        true_nn_modules = tx.copy_graphstate().output.nn_modules
        (false_r, false_graph, false_lifted_freevars) = speculate_branch(False)
        false_nn_modules = tx.copy_graphstate().output.nn_modules

        def dedup_and_sort_lifted_freevars(true_lifted_freevars, false_lifted_freevars):
            if False:
                return 10
            shared_freevars = true_lifted_freevars.keys() & false_lifted_freevars.keys()
            unique_true_freevars = true_lifted_freevars.keys() - shared_freevars
            unique_false_freevars = false_lifted_freevars.keys() - shared_freevars

            def _sort_by_name(vars):
                if False:
                    i = 10
                    return i + 15
                return sorted(vars, key=lambda var: var.node.name)
            return (list(_sort_by_name(list(shared_freevars))), list(_sort_by_name(list(unique_true_freevars))), list(_sort_by_name(list(unique_false_freevars))))
        (shared, unique_true, unique_false) = dedup_and_sort_lifted_freevars(true_lifted_freevars, false_lifted_freevars)

        def fixup_branch_inps(graph, lifted_freevars, shared, unique_true, unique_false):
            if False:
                print('Hello World!')

            def _insert_or_replace_phs(new_args, name_suffix):
                if False:
                    for i in range(10):
                        print('nop')
                for arg in new_args:
                    new_ph = graph.placeholder(arg.node.name + name_suffix)
                    if arg in lifted_freevars:
                        old_ph = lifted_freevars[arg].node
                        old_ph.replace_all_uses_with(new_ph)
                        old_ph.users = {}
                        graph.erase_node(old_ph)
            first_not_ph_node = next((node for node in graph.nodes if node.op != 'placeholder'))
            with graph.inserting_before(first_not_ph_node):
                _insert_or_replace_phs(shared, '')
                _insert_or_replace_phs(unique_true, '_true_branch')
                _insert_or_replace_phs(unique_false, '_false_branch')
        fixup_branch_inps(true_graph, true_lifted_freevars, shared, unique_true, unique_false)
        fixup_branch_inps(false_graph, false_lifted_freevars, shared, unique_true, unique_false)
        true_name = add_subgraph(tx, self.source, 'cond_true', torch.fx.GraphModule(true_nn_modules.nn_modules, true_graph))
        false_name = add_subgraph(tx, self.source, 'cond_false', torch.fx.GraphModule(false_nn_modules.nn_modules, false_graph))
        true_node = make_attr(tx, true_name)
        false_node = make_attr(tx, false_name)
        p_args = (args[0].as_proxy(), true_node, false_node, shared + unique_true + unique_false)
        example_value = true_r.as_proxy().node.meta['example_value']
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', torch.ops.higher_order.cond, args=tuple(p_args), kwargs={}), example_value=example_value)

def non_single_tensor_return_unsupported(api, ret):
    if False:
        print('Hello World!')
    from . import TensorVariable
    if not isinstance(ret, TensorVariable):
        raise Unsupported(f'{api} over function that returns something other than one Tensor')

class MapHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> VariableTracker:
        if False:
            for i in range(10):
                print('nop')
        from . import ConstantVariable, NestedUserFunctionVariable, TensorVariable, UserFunctionVariable
        from .builder import wrap_fx_proxy
        if len(kwargs) > 0:
            unimplemented('torch.ops.higher_order.map: kwargs are not supported in the map operator.')
        assert type(args[0].realize()) in (UserFunctionVariable, NestedUserFunctionVariable)
        assert type(args[1].realize()) is TensorVariable
        sample_shape = get_fake_value(args[1].as_proxy().node, tx).size()
        if len(sample_shape) < 1 or sample_shape[0] == 0:
            unimplemented("map() operator doesn't support scalar or zero-sized tensors during tracing.")
        checkpoint = tx.copy_graphstate()
        first_dim = args[1].call_method(tx, '__getitem__', args=[ConstantVariable.create(0)], kwargs={})
        ((body_r, _), body_graph, body_lifted_freevars) = speculate_subgraph(tx, args[0], [first_dim, *args[2:]], {}, tx.output.graph, checkpoint, 'torch.ops.higher_order.map', source_target=self.value)
        body_nn_modules = tx.copy_graphstate().output.nn_modules
        body_name = add_subgraph(tx, self.source, 'map_body', torch.fx.GraphModule(body_nn_modules.nn_modules, body_graph))
        body_node = make_attr(tx, body_name)
        p_args = (body_node, *(arg.as_proxy() for arg in args[1:]), *(arg for arg in body_lifted_freevars.keys()))
        non_single_tensor_return_unsupported('torch.ops.higher_order.map', body_r)
        r = body_r.as_proxy().node.meta['example_value']
        example_value = r.new_empty([sample_shape[0], *r.shape])
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)

class ExecutorchCallDelegateHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            for i in range(10):
                print('nop')
        from .builder import wrap_fx_proxy
        if len(kwargs) > 0:
            unimplemented('executorch_call_delegate: kwargs arguments were not enabled.')
        lowered_module = tx.output.get_submodule(args[0].module_key)
        lowered_node = make_attr(tx, args[0].module_key)
        p_args = tuple((arg.as_proxy() for arg in args[1:]))
        real_sub_args = pytree.tree_map_only(torch.fx.Proxy, lambda a: get_real_value(a.node, tx.output), p_args)
        example_res = lowered_module.original_module(*real_sub_args)
        example_value = deepcopy_to_fake_tensor(example_res, tx.fake_mode)
        p_args = (lowered_node,) + p_args
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)

class FunctorchGradHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            i = 10
            return i + 15
        from . import ConstantVariable
        from .builder import wrap_fx_proxy
        if not torch._dynamo.config.capture_func_transforms:
            unimplemented('torch.func.grad capture is disabled, it can be turned on by setting `torch._dynamo.config.capture_func_transforms=True`')
        checkpoint = tx.copy_graphstate()
        graph_checkpoint = tx.output.graph
        grad_args = (args[0], args[1], args[2])
        (func, argnums, has_aux) = grad_args
        kwargs = args[4].items
        if len(kwargs) > 0:
            unimplemented('torch.func.grad: kwargs arguments are currently unsupported.')
        ((body_r, _), body_graph, body_lifted_freevars) = speculate_subgraph(tx, func, args[3].items, {}, graph_checkpoint, checkpoint, 'torch.func.grad', source_target=self.value, enable_grad=True)
        body_name = add_subgraph(tx, self.source, 'grad_body', torch.fx.GraphModule(tx.output.nn_modules, body_graph))
        body_node = make_attr(tx, body_name)
        grad_proxy_args = (body_node, *(arg.as_proxy() for arg in grad_args[1:]))
        grad_fn = tx.output.create_proxy('call_function', torch.func.grad, args=tuple(grad_proxy_args), kwargs={}, name='grad_proxy')
        args = args[3].items
        grad_fn_args = tuple((arg.as_proxy() for arg in args)) + tuple(body_lifted_freevars)
        grad_output = grad_fn(*grad_fn_args)

        def _from_args(idx):
            if False:
                print('Hello World!')
            return args[idx].as_proxy().node.meta['example_value'].contiguous()

        def to_python_ints(argnums):
            if False:
                while True:
                    i = 10
            if not isinstance(argnums, (ConstantVariable, TupleVariable)):
                raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to be int or tuple of ints. Got {argnums}.')
            if isinstance(argnums, ConstantVariable):
                if not isinstance(argnums.value, (int, tuple)):
                    raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to be int or tuple of ints. Got {argnums}.')
                return argnums.value
            else:
                const_vars = argnums.unpack_var_sequence(tx)
                if not all((isinstance(var, ConstantVariable) and isinstance(var.value, int) for var in const_vars)):
                    raise UserError(UserErrorType.INVALID_INPUT, f'argnums is expected to contain int only. Got {const_vars}.')
                return tuple((var.value for var in const_vars))
        argnums_v = to_python_ints(argnums)
        example_value = pytree.tree_map(_from_args, argnums_v)
        if has_aux.value:
            body_r_proxy = body_r.as_proxy()
            aux = body_r_proxy[1].node.meta['example_value']
            example_value = (example_value, aux)
        fx_proxy = wrap_fx_proxy(tx=tx, proxy=grad_output, example_value=example_value)
        if not has_aux.value:
            if isinstance(argnums_v, int):
                return fx_proxy.call_method(tx, 'contiguous', (), {})
            else:
                grads = fx_proxy
                items = []
                for idx in range(len(argnums_v)):
                    proxy = grads.call_method(tx, '__getitem__', (ConstantVariable.create(idx),), {}).call_method(tx, 'contiguous', (), {})
                    items.append(proxy)
                return TupleVariable(items)
        else:
            grads = fx_proxy.call_method(tx, '__getitem__', (ConstantVariable.create(0),), {})
            aux = fx_proxy.call_method(tx, '__getitem__', (ConstantVariable.create(1),), {})
            if isinstance(argnums_v, int):
                return TupleVariable([grads.call_method(tx, 'contiguous', (), {}), aux])
            else:
                items = []
                for idx in range(len(argnums_v)):
                    proxy = grads.call_method(tx, '__getitem__', (ConstantVariable.create(idx),), {}).call_method(tx, 'contiguous', (), {})
                    items.append(proxy)
                return TupleVariable([TupleVariable(items), aux])

class FunctorchVmapHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            return 10
        from . import ConstantVariable, TensorVariable
        from .builder import wrap_fx_proxy
        if not torch._dynamo.config.capture_func_transforms:
            unimplemented('torch.func.vmap capture is disabled, it can be turned on by setting `torch._dynamo.config.capture_func_transforms=True`')
        checkpoint = tx.copy_graphstate()
        graph_checkpoint = tx.output.graph
        fn = args[0]
        in_dims = args[1]
        out_dims = args[2]
        randomness = args[3]
        chunk_size = args[4]
        batch_input_args = args[5:]
        if not isinstance(in_dims, (ConstantVariable, TupleVariable)):
            unimplemented('torch.func.vmap: in_dims is not an int or tuple variable.')
        if not isinstance(out_dims, (ConstantVariable, TupleVariable)):
            unimplemented('torch.func.vmap: out_dims is not an int or tuple variable.')
        if len(kwargs) > 0:
            unimplemented('NYI - torch.func.vmap: kwargs arguments are currently unsupported.')
        if chunk_size.value is not None:
            unimplemented('NYI - torch.func.vmap is not implemented when chunk_size is passed')
        tree_flatten = UserFunctionVariable(pytree.tree_flatten)
        (flat_args, arg_spec) = tree_flatten.call_function(tx, [ListVariable(batch_input_args)], {}).unpack_var_sequence(tx)
        in_dims_v = in_dims if isinstance(in_dims.as_python_constant(), int) else BuiltinVariable(list).call_function(tx, [in_dims], {})
        broadcast_to_and_flatten = UserFunctionVariable(pytree._broadcast_to_and_flatten)
        broadcasted_in_dims = broadcast_to_and_flatten.call_function(tx, [in_dims_v, arg_spec], {})
        unbatched_input_args = []
        for (arg, in_dim) in zip(flat_args.unpack_var_sequence(tx), broadcasted_in_dims.unpack_var_sequence(tx)):
            if in_dim is not None:
                assert isinstance(arg, TensorVariable)
                unbatched_arg = arg.call_method(tx, 'select', [in_dim, ConstantVariable.create(0)], {})
                unbatched_input_args.append(unbatched_arg)
            else:
                unbatched_input_args.append(arg)
        tree_unflatten = UserFunctionVariable(pytree.tree_unflatten)
        with tx.strict_translation_mode():
            (_, body_graph, body_lifted_freevars) = speculate_subgraph(tx, fn, tree_unflatten.call_function(tx, [ListVariable(unbatched_input_args), arg_spec], {}).unpack_var_sequence(tx), {}, graph_checkpoint, checkpoint, 'torch.vmap', source_target=self.value)
        body_name = add_subgraph(tx, self.source, 'vmap_body', torch.fx.GraphModule(tx.output.nn_modules, body_graph))
        body_node = make_attr(tx, body_name)
        updated_in_dims = TupleVariable(broadcasted_in_dims.unpack_var_sequence(tx) + [ConstantVariable.create(None)] * len(body_lifted_freevars))
        vmap_proxy_args = (body_node, *(arg.as_proxy() for arg in (updated_in_dims, out_dims, randomness)))
        vmap_proxy = tx.output.create_proxy('call_function', torch.func.vmap, args=tuple(vmap_proxy_args), kwargs={}, name='vmap_proxy')
        proxy_batched_fn_args = tuple((arg.as_proxy() for arg in batch_input_args)) + tuple(body_lifted_freevars)
        fake_batched_fn_args = itertools.chain((get_fake_value(arg.as_proxy().node, tx) for arg in batch_input_args), (get_fake_value(arg.node, tx) for arg in body_lifted_freevars))
        actual_in_dims = tuple(pytree.tree_map(lambda x: x.value, updated_in_dims.items))
        with tx.fake_mode, enable_python_dispatcher():
            example_value = torch._functorch.vmap.vmap_impl(torch.fx.GraphModule(tx.output.nn_modules, body_graph), actual_in_dims, out_dims.as_python_constant(), randomness.value, chunk_size.value, *fake_batched_fn_args)
        proxy = vmap_proxy(*proxy_batched_fn_args)
        return wrap_fx_proxy(tx=tx, proxy=proxy, example_value=example_value)

class AutogradFunctionMethodHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def __init__(self, value, fwd_bwd_tracer=None, source: Optional[Source]=None, **kwargs):
        if False:
            print('Hello World!')
        super().__init__(value, source, **kwargs)
        self.fwd_bwd_tracer = fwd_bwd_tracer

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            return 10
        from . import UserFunctionVariable
        from .builder import wrap_fx_proxy
        tracer = self.fwd_bwd_tracer
        if len(kwargs) > 0:
            unimplemented('kwargs have not been implemented for torch.autograd.Function')
        from . import TorchVariable
        always_restore = self.value.__name__ == 'trampoline_autograd_bwd'
        if self.value.__name__ == 'trampoline_autograd_bwd' or self.value.__name__ == 'trampoline_autograd_fwd':
            fn = UserFunctionVariable(self.value, source=self.source)
        else:
            fn = TorchVariable(self.value)
        checkpoint = tx.copy_graphstate()
        pre_guards = tx.output.guards
        graph_checkpoint = tx.output.graph
        ((body_r, _), body_graph, body_lifted_freevars) = speculate_subgraph(tx, fn, [*args], {}, graph_checkpoint, checkpoint, 'the user-defined autograd.Function', source_target=self.value, always_restore=always_restore, restore_side_effects=False, tracer=tracer)
        post_guards = tx.output.guards
        if body_lifted_freevars:
            unimplemented('NYI - freevars in autograd function.')
        if always_restore:
            if post_guards - pre_guards:
                unimplemented('NYI - New guards discovered in a restoring state')
            return None
        p_args = (*(arg.as_proxy() for arg in args), *(arg for arg in body_lifted_freevars.keys()))
        example_value = pytree.tree_map_only(torch.fx.Proxy, lambda a: a.node.meta['example_value'], body_r.as_proxy())
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)

class WrapHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def create_wrapped_node(self, tx, args, kwargs, description):
        if False:
            i = 10
            return i + 15
        checkpoint = tx.copy_graphstate()
        graph_checkpoint = tx.output.graph
        ((body_r, treespec), body_graph, body_lifted_freevars) = speculate_subgraph(tx, args[0], [*args[1:]], kwargs, graph_checkpoint, checkpoint, description, source_target=self.value, manually_set_subgraph_inputs=False, should_flatten_outputs=True)
        body_gmod = torch.fx.GraphModule(tx.output.nn_modules, body_graph)
        body_name = add_subgraph(tx, self.source, 'wrap_body', body_gmod)
        body_node = make_attr(tx, body_name)
        lifted_args = tuple((arg for arg in body_lifted_freevars.keys()))
        proxy_args = (body_node,) + lifted_args
        example_value = pytree.tree_map_only(torch.fx.Proxy, lambda a: a.node.meta['example_value'], body_r.as_proxy())
        return (proxy_args, {}, example_value, treespec, body_gmod)

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            while True:
                i = 10
        from .builder import wrap_fx_proxy
        (p_args, p_kwargs, example_value, treespec, _) = self.create_wrapped_node(tx, args, kwargs, 'wrap')
        if len(p_kwargs) > 0:
            unimplemented('kwargs should have been flattened into lifted args')
        variable = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)
        if treespec is None:
            return variable
        variable = BuiltinVariable(list).call_function(tx, [variable], {})
        tree_unflatten = UserFunctionVariable(pytree.tree_unflatten)
        return tree_unflatten.call_function(tx, [variable, treespec], {})

class OutDtypeHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            for i in range(10):
                print('nop')
        from .builder import wrap_fx_proxy
        if len(kwargs) > 0:
            unimplemented('out_dtype does not handle kwargs')
        p_args = tuple((arg.as_proxy() for arg in args))
        op = p_args[0]
        output_dtype = p_args[1]
        fake_sub_args = pytree.tree_map_only(torch.fx.Proxy, lambda a: a.node.meta['example_value'], p_args[2:])
        example_value = op(*fake_sub_args).to(dtype=output_dtype)
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs={}), example_value=example_value)

class CheckpointHigherOrderVariable(WrapHigherOrderVariable):

    def call_function(self, tx, args: List[VariableTracker], kwargs: Dict[str, VariableTracker]) -> VariableTracker:
        if False:
            print('Hello World!')
        from torch._higher_order_ops.wrap import TagActivationCheckpoint
        from torch.utils.checkpoint import noop_context_fn
        from .builder import wrap_fx_proxy
        context_fn = None
        if 'context_fn' in kwargs and kwargs['context_fn'] != noop_context_fn:
            context_fn = kwargs.pop('context_fn').fn
        (checkpoint_kwargs, gmod_kwargs) = TagActivationCheckpoint.divide_kwargs(kwargs)
        (p_args, _, example_value, treespec, checkpointed_gmod) = self.create_wrapped_node(tx, args, gmod_kwargs, 'torch.utils.checkpoint.checkpoint')
        if context_fn is not None:
            checkpointed_gmod.meta['_checkpoint_context_fn'] = context_fn
        (_, checkpoint_kwargs) = proxy_args_kwargs([], checkpoint_kwargs)
        variable = wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=tuple(p_args), kwargs=checkpoint_kwargs), example_value=example_value)
        if treespec is None:
            return variable
        variable = BuiltinVariable(list).call_function(tx, [variable], {})
        tree_unflatten = UserFunctionVariable(pytree.tree_unflatten)
        return tree_unflatten.call_function(tx, [variable, treespec], {})

class ExportTracepointHigherOrderVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            return 10
        from .builder import wrap_fx_proxy
        p_args = tuple((arg.as_proxy() for arg in args))
        p_kwargs = {key: arg.as_proxy() for (key, arg) in kwargs.items()}
        return wrap_fx_proxy(tx=tx, proxy=tx.output.create_proxy('call_function', self.value, args=p_args, kwargs=p_kwargs), example_value=None)

class TraceWrappedHigherOrderOperatorVariable(TorchHigherOrderOperatorVariable):

    def call_function(self, tx, args: 'List[VariableTracker]', kwargs: 'Dict[str, VariableTracker]') -> 'VariableTracker':
        if False:
            print('Hello World!')
        from . import TensorVariable
        assert 'fn' in kwargs
        fn = kwargs['fn']
        assert len(args) == 1
        grad = args[0]
        assert isinstance(grad, TensorVariable)
        return fn.call_function(tx, args, {})