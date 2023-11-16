"""Utilities for V2 control flow."""
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.eager import context
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import concrete_function
from tensorflow.python.eager.polymorphic_function import tracing_compilation
from tensorflow.python.eager.polymorphic_function import transform
from tensorflow.python.framework import function_def_to_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework.func_graph import FuncGraph
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_v2_func_graphs
from tensorflow.python.ops import gradients_util
from tensorflow.python.util import keras_deps
from tensorflow.python.util import tf_contextlib
from tensorflow.python.util.tf_export import tf_export
_EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE = None
_DISABLE_LOWER_USING_SWITCH_MERGE = False
CondBranchFuncGraph = control_flow_v2_func_graphs.CondBranchFuncGraph
WhileCondFuncGraph = control_flow_v2_func_graphs.WhileCondFuncGraph
WhileBodyFuncGraph = control_flow_v2_func_graphs.WhileBodyFuncGraph

def in_defun():
    if False:
        i = 10
        return i + 15
    'Returns if the current graph is, or is nested in, a defun.'
    if context.executing_eagerly():
        return False
    graph = ops.get_default_graph()
    while isinstance(graph, CondBranchFuncGraph) or isinstance(graph, WhileBodyFuncGraph) or isinstance(graph, WhileCondFuncGraph):
        graph = graph.outer_graph
    return isinstance(graph, FuncGraph)

def in_while_loop_defun(graph):
    if False:
        for i in range(10):
            print('nop')
    'Returns if the graph is a while loop FuncGraph.'
    if context.executing_eagerly():
        return False
    return isinstance(graph, WhileCondFuncGraph) or isinstance(graph, WhileBodyFuncGraph)

def create_new_tf_function(func_graph):
    if False:
        i = 10
        return i + 15
    'Converts func_graph to a TF_Function and adds it to the current graph.\n\n  Args:\n    func_graph: FuncGraph\n\n  Returns:\n    The name of the new TF_Function.\n  '
    transform.apply_func_graph_transforms(func_graph)
    func = atomic_function.from_func_graph(func_graph.name, func_graph, {})
    func_graph.outer_graph._add_function_recursive(func)
    return func_graph.name

def unique_fn_name(scope, name):
    if False:
        print('Hello World!')
    'Returns a unique name to use for a control flow function.\n\n  Args:\n    scope: A name scope string.\n    name: An identifier for this function (e.g. "true", "body").\n\n  Returns:\n    A string, the name to use for the function.\n  '
    return ('%s%s_%s' % (scope, name, ops.uid())).replace('/', '_')

def unique_grad_fn_name(forward_name):
    if False:
        return 10
    return '%s_grad_%s' % (forward_name, ops.uid())

def maybe_set_lowering_attr(op, lower_using_switch_merge=None):
    if False:
        i = 10
        return i + 15
    "Sets the flag to enable lowering on `op` if necessary.\n\n  Lowering allows cond_v2 and while_v2 to avoid some of the limitations of\n  Functions, allowing users to specify devices & colocation inside of cond_v2\n  and while_v2 input functions, and enabling non-strict evaluation & partial\n  pruning. This brings v2 control flow closer to feature parity with v1 control\n  flow.\n\n  However, we do not lower in the following cases:\n    - When the `If` or `While` ops are in the XLA context. Because it is easier\n      for XLA to apply its own optimizations when dealing with un-lowered\n      control flow operators than with low-level control flow primitives.\n    - When the eager execution context specifies the executor of functions to\n      be the single threaded executor (see context.function_executor_type()).\n      Because the single threaded executor does not support v1 control flow ops.\n    - When 'lower_using_switch_merge' is explicitly set to False.\n\n  Args:\n    op: An `If` or `While` Operation.\n    lower_using_switch_merge: Explicit value to lower or not (optional).\n  "
    if lower_using_switch_merge is not None:
        op._set_attr('_lower_using_switch_merge', attr_value_pb2.AttrValue(b=lower_using_switch_merge))
    elif not _DISABLE_LOWER_USING_SWITCH_MERGE and (not control_flow_util.GraphOrParentsInXlaContext(op.graph)) and (context.context().function_call_options.executor_type != 'SINGLE_THREADED_EXECUTOR'):
        op._set_attr('_lower_using_switch_merge', attr_value_pb2.AttrValue(b=True))

def maybe_propagate_compile_time_consts_in_xla(op):
    if False:
        print('Hello World!')
    "Tells XLA whether to propagate compile-time consts in the loop body.\n\n  This is needed to make compile time constants available to ops, for example\n  `max_num_elements` in `EmptyTensorList`, inside the loop body. Ideally this\n  would always be turned on, but that doesn't work with legacy functionalized\n  while_loops.\n\n  Args:\n    op: A `While` Operation.\n  "
    if control_flow_util.GraphOrParentsInXlaContext(op.graph):
        op._set_attr('_xla_propagate_compile_time_consts', attr_value_pb2.AttrValue(b=True))

def resource_input_index(tensor_name, input_names, node_defs, functions):
    if False:
        print('Hello World!')
    'Returns the index of the input corresponding to `tensor_name`.\n\n  This method is used to find the corresponding index of an arbitrary resource\n  tensor in a function (the function could be a loop body). We assume that\n  resource handles are never created in functions, so that every resource\n  tensor can be traced back to a function input.\n\n  The awkward signature of this method is to make it work with both FuncGraphs\n  and FunctionDefs. This is so we can recurse on function call ops without\n  building the corresponding FuncGraph (note that even if a FuncGraph for a\n  FunctionDef already exists, the input/output/node names may have been\n  changed when the FuncGraph was serialized to the FunctionDef, which makes it\n  unusable with this algorithm).\n\n  Args:\n    tensor_name: the name of the resource tensor to be resolved to an input.\n    input_names: a list of the names of all inputs to the function.\n    node_defs: a dict mapping op name -> NodeDef for every op in the function.\n    functions: a dict mapping function name -> AtomicFunction.\n\n  Returns:\n    The index into input_names corresponding to `tensor_name`.\n  '
    while tensor_name not in input_names:
        parts = tensor_name.split(':')
        if len(parts) == 3:
            (op_name, _, output_idx) = parts
        elif len(parts) == 2:
            (op_name, output_idx) = parts
        else:
            assert len(parts) == 1
            op_name = parts[0]
            output_idx = 0
            tensor_name = '%s:%d' % (tensor_name, output_idx)
            if tensor_name in input_names:
                break
        output_idx = int(output_idx)
        node_def = node_defs[op_name]

        def _extract_input_index(function_attribute_name):
            if False:
                return 10
            func_name = node_def.attr[function_attribute_name].func.name
            fdef = functions[func_name].cached_definition
            output_arg_name = fdef.signature.output_arg[output_idx].name
            output_tensor_name = fdef.ret[output_arg_name]
            return resource_input_index(output_tensor_name, [arg.name for arg in fdef.signature.input_arg], {ndef.name: ndef for ndef in fdef.node_def}, functions)
        if node_def.op in ('Identity', 'While'):
            tensor_name = node_def.input[output_idx]
        elif node_def.op in ('PartitionedCall', 'StatefulPartitionedCall'):
            tensor_name = node_def.input[_extract_input_index('f')]
        elif node_def.op in ('If', 'StatelessIf'):
            input_index = _extract_input_index('then_branch')
            if input_index != _extract_input_index('else_branch'):
                raise AssertionError('Expected cond branches ({} op) to each have the same input->output mapping of resources.'.format(node_def.op))
            tensor_name = node_def.input[input_index + 1]
        else:
            raise ValueError('Taking gradient of a while loop which creates a resource in its body is not supported: %s (%s)' % (op_name, node_def.op))
    return input_names.index(tensor_name)

@tf_contextlib.contextmanager
def clear_control_inputs():
    if False:
        return 10
    'Clears the control inputs but preserves the ControlFlowContext.\n\n  This is needed to preserve the XLAControlFlowControl when clearing\n  control inputs for the gradient accumulators in while_v2.\n  `ops.control_dependencies` does not allow that.\n\n  Yields:\n    A context manager in which the ops created will not have any control inputs\n    by default but the control flow context is the same.\n  '
    control_flow_context = ops.get_default_graph()._get_control_flow_context()
    with ops.control_dependencies(None):
        ops.get_default_graph()._set_control_flow_context(control_flow_context)
        yield

def _is_tpu_strategy(strategy):
    if False:
        while True:
            i = 10
    return strategy is not None and strategy.__class__.__name__.startswith('TPUStrategy')

def _is_building_keras_layer():
    if False:
        i = 10
        return i + 15
    keras_call_context_function = keras_deps.get_call_context_function()
    if keras_call_context_function:
        return keras_call_context_function().layer is not None
    else:
        return False

def output_all_intermediates():
    if False:
        for i in range(10):
            print('nop')
    'Whether to output all intermediates of a functional control flow op.\n\n  The default behavior is to output intermediates only when building a Keras\n  Layer in graph mode and that too when certain other conditions are met:\n  1. We do not output intermediates if the functional control flow op\n     is being built inside a FuncGraph which is not a If/While graph. This\n     guards against outputting intermediates in eager mode since keras adds\n     tensors to a FuncGraph named "keras_graph" in that case. Also because we\n     do not output intermediates of tf.function (since this feature is only for\n     backwards compatibility) outputting intermediates of functional control\n     flow ops built inside tf.function is of no value.\n  2. We do not output intermediates when the compilation is using XLA or for a\n     TPU.\n  3. We do not output intermediates when a single threaded executor is used\n     since that does not perform inlining and pruning.\n\n  Returns:\n    A bool telling whether to output all intermediates.\n  '
    if _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE is not None:
        return _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    if in_defun():
        return False
    if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
        return False
    if context.context().function_call_options.executor_type == 'SINGLE_THREADED_EXECUTOR':
        return False
    return _is_building_keras_layer()

def get_func_graph(op, input_shapes, func_name):
    if False:
        for i in range(10):
            print('nop')
    'Generates and returns a FuncGraph for the given op and input_shapes.'
    fdef = None
    graph = op.graph
    while graph is not None:
        func = graph._get_function(func_name)
        if func is not None:
            fdef = func.cached_definition
            break
        if hasattr(graph, 'outer_graph'):
            graph = graph.outer_graph
        else:
            break
    if fdef is None:
        raise KeyError('%s cannot be found in the graph' % func_name)
    with op.graph.as_default():
        func_graph = function_def_to_graph.function_def_to_graph(fdef, input_shapes=input_shapes)
    for operation in func_graph.get_operations():
        if operation.type in ['PartitionedCall', 'StatefulPartitionedCall']:
            f = graph._get_function(operation.get_attr('f').name)
            try:
                cf = concrete_function.ConcreteFunction.from_func_graph(f.graph, f.function_type, attrs=f.cached_definition.attr)
            except AttributeError:
                continue
            operation._gradient_function = cf._get_gradient_function()
    return func_graph

def get_op_and_outputs(op_or_outputs):
    if False:
        print('Hello World!')
    if isinstance(op_or_outputs, ops.Operation):
        return (op_or_outputs, [])
    elif not op_or_outputs:
        return (None, [])
    else:
        return (op_or_outputs[0].op, op_or_outputs)

def graph_wrapped_for_higher_order_tape_gradients(graph):
    if False:
        for i in range(10):
            print('nop')
    'Check if `graph` is wrapped by `run_as_function_for_tape_gradients`.'
    while graph is not None:
        if 'cflow_gradient_wrapper' in getattr(graph, 'name', ''):
            return True
        graph = getattr(graph, 'outer_graph', None)
    return False

def run_as_function_for_tape_gradients(make_op, inputs):
    if False:
        return 10
    "Fix higher-order tape gradients by wrapping `make_op` in a function.\n\n  Args:\n    make_op: A function that takes a list of inputs and returns a list of output\n      tensors. This function should set any handle data relevant to its outputs\n      before returning.\n    inputs: A list of tensors to check for tape gradients and pass to\n      `make_op`. These should include all tensors used in `make_op`.\n\n  Returns:\n    Tensors corresponding to `make_op`'s output.\n  "
    if gradients_util.PossibleTapeGradientTypes(inputs) == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER and (not (ops.get_default_graph().building_function and 'cflow_gradient_wrapper' in ops.get_default_graph().name)):
        results = tracing_compilation.call_function((inputs,), tracing_options=tracing_compilation.TracingOptions(make_op, 'cflow_gradient_wrapper', autograph=False))
        return results
    else:
        return make_op(inputs)

@tf_export(v1=['experimental.output_all_intermediates'])
def set_output_all_intermediates(state):
    if False:
        i = 10
        return i + 15
    'Whether to output all intermediates from functional control flow ops.\n\n  The "default" behavior to is to output all intermediates when using v2 control\n  flow inside Keras models in graph mode (possibly inside Estimators). This is\n  needed to support taking gradients of v2 control flow. In graph mode, Keras\n  can sometimes freeze the forward graph before the gradient computation which\n  does not work for v2 control flow since it requires updating the forward ops\n  to output the needed intermediates. We work around this by proactively\n  outputting the needed intermediates when building the forward pass itself.\n  Ideally any such extra tensors should be pruned out at runtime. However, if\n  for any reason this doesn\'t work for you or if you have an inference-only\n  model you can turn this behavior off using\n  `tf.compat.v1.experimental.output_all_intermediates(False)`.\n\n  If with the default behavior you are still seeing errors of the form\n  "Connecting to invalid output X of source node Y which has Z outputs" try\n  setting `tf.compat.v1.experimental.output_all_intermediates(True)` and\n  please file an issue at https://github.com/tensorflow/tensorflow/issues.\n\n  Args:\n    state: True, False or None. None restores the default behavior.\n  '
    global _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE
    _EXPERIMENTAL_OUTPUT_ALL_INTERMEDIATES_OVERRIDE = state