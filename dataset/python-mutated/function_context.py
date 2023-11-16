"""Context information for a tf.function."""
from typing import NamedTuple, Any
from tensorflow.core.function.polymorphism import function_cache
from tensorflow.python.eager import context
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.saved_model import save_context

class EagerContext(NamedTuple):
    parent_graph: Any
    device_functions: Any
    colocation_stack: Any
    in_cross_replica_context: Any
    variable_policy: Any
    xla_context_id: Any

def make_function_context(scope_type=None) -> function_cache.FunctionContext:
    if False:
        for i in range(10):
            print('nop')
    'Generates a FunctionContext based on current contextual info.'
    ctx = context.context()
    executing_eagerly = ctx.executing_eagerly()
    parent_graph = None
    xla_context_id = 0
    if not executing_eagerly:
        xla_context = _enclosing_xla_context()
        if xla_context is not None and xla_context.RequiresUniqueFunctionRetracing():
            xla_context_id = id(xla_context)
        with ops.init_scope():
            executing_eagerly = ctx.executing_eagerly()
            parent_graph = None if executing_eagerly else ops.get_default_graph()
    default_graph = ops.get_default_graph()
    strategy_stack = default_graph._distribution_strategy_stack
    uses_distribution_strategy = strategy_stack and strategy_stack[-1].strategy.extended._retrace_functions_for_each_device
    if executing_eagerly:
        colocation_stack = ()
        if uses_distribution_strategy:
            device_functions = (pydev.merge_device(ctx.device_name),)
        else:
            device_functions = ()
    else:
        colocation_stack = tuple(default_graph._colocation_stack.peek_objs())
        if uses_distribution_strategy or func_graph_module.device_stack_has_callable(default_graph._device_function_stack):
            device_functions = tuple(default_graph._device_functions_outer_to_inner)
        else:
            device_functions = ()
    in_cross_replica_context = False
    try:
        in_cross_replica_context = strategy_stack[-1].replica_context is None
    except (AttributeError, IndexError):
        pass
    if save_context.in_save_context():
        variable_policy = save_context.get_save_options().experimental_variable_policy
    else:
        variable_policy = None
    return function_cache.FunctionContext(EagerContext(parent_graph, device_functions, colocation_stack, in_cross_replica_context, variable_policy, xla_context_id), scope_type)

def _enclosing_xla_context():
    if False:
        return 10
    'Returns the XLAControlFlowContext, which exists inside a tpu.rewrite().'
    graph = ops.get_default_graph()
    while graph is not None:
        context_ = graph._get_control_flow_context()
        while context_ is not None:
            if isinstance(context_, control_flow_ops.XLAControlFlowContext):
                return context_
            context_ = context_.outer_context
        graph = getattr(graph, 'outer_graph', None)
    return None