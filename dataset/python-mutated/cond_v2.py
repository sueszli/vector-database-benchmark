"""cond_v2 and gradient.

This is a version of cond that emits a single If op, as well as the gradient
function for If ops produced by cond_v2. This will eventually replace the
current tf.cond implementation once it reaches feature and performance parity.
"""
import collections
from tensorflow.core.framework import types_pb2
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import none_tensor
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_util
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_optional_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import math_ops
from tensorflow.python.util import nest
_COND = 1
_CASE = 2

def cond_v2(pred, true_fn, false_fn, name='cond'):
    if False:
        for i in range(10):
            print('nop')
    'Like tf.cond, except emits a single If op.'
    if isinstance(pred, bool):
        raise TypeError('pred must not be a Python bool', pred)
    if not name:
        name = 'cond'
    with ops.name_scope(name) as scope:
        true_name = util.unique_fn_name(scope, 'true')
        false_name = util.unique_fn_name(scope, 'false')
        add_control_dependencies = ops.get_default_graph()._add_control_dependencies
        pred = ops.convert_to_tensor(pred)
        if tensor_util.is_tf_type(pred) and (pred.shape.dims is None or pred.shape.dims):
            pred = array_ops.squeeze_v2(pred)
        true_graph = func_graph_module.func_graph_from_py_func(true_name, true_fn, [], {}, func_graph=util.CondBranchFuncGraph(true_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies, op_return_value=pred)
        false_graph = func_graph_module.func_graph_from_py_func(false_name, false_fn, [], {}, func_graph=util.CondBranchFuncGraph(false_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies, op_return_value=pred)
        verify_captures(_COND, [true_graph, false_graph])
        return _build_cond(pred, true_graph, false_graph, true_graph.external_captures, false_graph.external_captures, building_gradient=False, name=scope)

@ops.RegisterGradient('StatelessIf')
@ops.RegisterGradient('If')
def _IfGrad(op, *grads):
    if False:
        for i in range(10):
            print('nop')
    'The gradient of an If op produced by cond_v2.'
    if_op = op.outputs[0].op
    (true_graph, false_graph) = get_func_graphs(if_op)
    assert true_graph.outer_graph == if_op.graph
    assert false_graph.outer_graph == if_op.graph
    true_grad_graph = _create_grad_func(true_graph, grads, util.unique_grad_fn_name(true_graph.name))
    false_grad_graph = _create_grad_func(false_graph, grads, util.unique_grad_fn_name(false_graph.name))
    _create_zeros_for_none_grads([true_graph, false_graph], [true_grad_graph, false_grad_graph])
    if true_grad_graph.op_needs_rewrite or false_grad_graph.op_needs_rewrite:
        if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
            true_intermediates = true_grad_graph.xla_intermediates
            false_intermediates = false_grad_graph.xla_intermediates
            (extra_true_outputs, extra_false_outputs) = _make_intermediates_match_xla([true_graph, false_graph], [true_intermediates, false_intermediates])
        else:
            true_intermediates = true_grad_graph.wrapped_intermediates
            false_intermediates = false_grad_graph.wrapped_intermediates
            (extra_true_outputs, extra_false_outputs) = _make_intermediates_match([true_graph, false_graph], [true_intermediates, false_intermediates])
        true_graph.outputs.extend(extra_true_outputs)
        false_graph.outputs.extend(extra_false_outputs)
        _check_same_outputs(_COND, [true_graph, false_graph])
        true_graph.name += '_rewritten'
        false_graph.name += '_rewritten'
        if_op._set_func_attr('then_branch', util.create_new_tf_function(true_graph))
        if_op._set_func_attr('else_branch', util.create_new_tf_function(false_graph))
        if_op._set_type_list_attr('Tout', true_graph.output_types)
        if_op._set_shape_list_attr('output_shapes', true_graph.output_shapes)
        if_op._add_outputs([t.dtype for t in extra_true_outputs], [t.shape for t in extra_true_outputs])
    true_grad_inputs = _resolve_grad_inputs(true_graph, true_grad_graph)
    false_grad_inputs = _resolve_grad_inputs(false_graph, false_grad_graph)
    _make_output_composite_tensors_match(_COND, [true_grad_graph, false_grad_graph])
    outputs = _build_cond(if_op.inputs[0], true_grad_graph, false_grad_graph, true_grad_inputs, false_grad_inputs, building_gradient=True)
    return [None] + outputs

def _is_op_stateful(op):
    if False:
        while True:
            i = 10
    'Check whether an op is stateful.\n\n  This helper function handles two special cases to make the stateful analysis\n  consistent with the mlir side effect analysis.\n  1. GlobalIterIdOp should be stateless.\n  2. CollectiveGatherV2 with attribute is_stateless to be True should be\n     stateless.\n\n  Args:\n   op: Operation\n\n  Returns:\n    Boolean indicates whether the operation is stateless or not.\n  '
    if op.type == 'GlobalIterId':
        return False
    if op.type == 'CollectiveGatherV2' and op.get_attr('is_stateless'):
        return False
    return op._is_stateful

def _build_cond(pred, true_graph, false_graph, true_inputs, false_inputs, building_gradient, name=None):
    if False:
        print('Hello World!')
    "Creates an If op from the specified predicate, branch functions and inputs.\n\n  Note that this modifies true_graph and false_graph to make the inputs match,\n  and to output all intermediates values so they're available for the gradient\n  computation.\n\n  true_graph and false_graph need not have the same input types, but they must\n  have the same output types.\n\n  Args:\n    pred: boolean Tensor\n    true_graph: FuncGraph\n    false_graph: FuncGraph\n    true_inputs: a list of Tensors to be passed to true_graph as input.\n    false_inputs: a list of Tensors to be passed to false_graph as input.\n    building_gradient: Whether this is a gradient If op.\n    name: the name for the If op.\n\n  Returns:\n    A list of Tensors which are the outputs of the If op. Does not include added\n    intermediate outputs.\n  "
    _make_indexed_slices_indices_types_match(_COND, [true_graph, false_graph])
    _check_same_outputs(_COND, [true_graph, false_graph])
    cond_inputs = _make_inputs_match([true_graph, false_graph], [true_inputs, false_inputs])
    if not building_gradient and util.output_all_intermediates():
        true_intermediates = _get_intermediates(true_graph)
        false_intermediates = _get_intermediates(false_graph)
        wrapped_true_intermediates = _wrap_intermediates(true_graph, true_intermediates)
        wrapped_false_intermediates = _wrap_intermediates(false_graph, false_intermediates)
        (extra_true_outputs, extra_false_outputs) = _make_intermediates_match([true_graph, false_graph], [wrapped_true_intermediates, wrapped_false_intermediates])
        true_graph.outputs.extend(extra_true_outputs)
        false_graph.outputs.extend(extra_false_outputs)
        _check_same_outputs(_COND, [true_graph, false_graph])
    with ops.control_dependencies(list(true_graph.function_captures.control) + list(false_graph.function_captures.control)):
        true_stateful_ops = [op for op in true_graph.get_operations() if _is_op_stateful(op)]
        false_stateful_ops = [op for op in false_graph.get_operations() if _is_op_stateful(op)]
        if true_stateful_ops or false_stateful_ops:
            op_fn = gen_functional_ops._if
        else:
            op_fn = gen_functional_ops.stateless_if

        def _make_op(inputs):
            if False:
                print('Hello World!')
            (if_op, tensors) = util.get_op_and_outputs(op_fn(pred, inputs, [t.dtype for t in true_graph.outputs], util.create_new_tf_function(true_graph), util.create_new_tf_function(false_graph), output_shapes=_get_output_shapes(true_graph.outputs, false_graph.outputs), name=name))
            _copy_handle_data(tensors, true_graph.outputs, false_graph.outputs)
            if if_op is not None:
                true_graph.outer_graph = ops.get_default_graph()
                false_graph.outer_graph = ops.get_default_graph()
                if_op._true_graph = true_graph
                if_op._false_graph = false_graph
                util.maybe_set_lowering_attr(if_op)
                util.maybe_propagate_compile_time_consts_in_xla(if_op)
                _set_read_only_resource_inputs_attr(if_op, [true_graph, false_graph])
                if_op.graph.prevent_fetching(if_op)
            return tensors
        tensors = util.run_as_function_for_tape_gradients(_make_op, cond_inputs)
    tensors = [array_ops.identity(t) for t in tensors]
    structured_output_specs = _get_compatible_structured_output_specs(true_graph, false_graph)
    return _pack_sequence_as(structured_output_specs, tensors)

def get_func_graphs(op):
    if False:
        return 10
    'Returns `FuncGraph`s for the input op branches.\n\n  Args:\n    op: The If or Case Operation.\n\n  Returns:\n    A tuple of the `FuncGraph`s of the then_branch and else_branch (all branches\n    for Case).\n  '

    def _get_func_graph_for_branch(name_attr_list, cached_attr_name=None):
        if False:
            return 10
        'Generates and returns a FuncGraph for the given branch.'
        func_graph = None
        if cached_attr_name is not None:
            func_graph = getattr(op, cached_attr_name, None)
        inputs = op.inputs[1:]
        if func_graph is None:
            input_shapes = [t.shape for t in inputs]
            func_graph = util.get_func_graph(op, input_shapes, name_attr_list.name)
        for (external_t, internal_t) in zip(inputs, func_graph.inputs):
            handle_data_util.copy_handle_data(external_t, internal_t)
        func_graph.function_captures.reset_captures(inputs, func_graph.inputs)
        func_graph._forward_cond = op
        return func_graph
    if op.type in ['If', 'StatelessIf']:
        return (_get_func_graph_for_branch(op.get_attr('then_branch'), '_true_graph'), _get_func_graph_for_branch(op.get_attr('else_branch'), '_false_graph'))
    elif op.type in ['Case', 'StatelessCase']:
        return [_get_func_graph_for_branch(branch_fn, '_branch_graph_{}'.format(i)) for (i, branch_fn) in enumerate(op.get_attr('branches'))]
    else:
        raise ValueError('Unsupported op type: {}'.format(op.type))

def _get_compatible_structured_output_specs(true_graph, false_graph):
    if False:
        for i in range(10):
            print('nop')
    'Returns the most specific compatible specs of graph structured outputs.'
    return nest.map_structure(_get_compatible_spec, true_graph.structured_outputs, false_graph.structured_outputs)

def _get_compatible_spec(value_or_spec1, value_or_spec2):
    if False:
        print('Hello World!')
    'Returns the most specific compatible spec.\n\n  Args:\n    value_or_spec1: A TypeSpecs or a value that has a defined TypeSpec.\n    value_or_spec2: A TypeSpecs or a value that has a defined TypeSpec.\n\n  Returns:\n    The most specific compatible TypeSpecs of the input.\n\n  Raises:\n    ValueError: If value_or_spec1 is not compatible with value_or_spec2.\n  '
    spec1 = _get_spec_for(value_or_spec1)
    spec2 = _get_spec_for(value_or_spec2)
    common = spec1._without_tensor_names().most_specific_common_supertype([spec2._without_tensor_names()])
    if common is None:
        raise TypeError(f'No common supertype of {spec1} and {spec2}.')
    return common

def _get_spec_for(value_or_spec):
    if False:
        return 10
    'Returns TypeSpec of a value or itself if it is a TypeSpec already.'
    if isinstance(value_or_spec, type_spec.TypeSpec):
        return value_or_spec
    return type_spec.type_spec_from_value(value_or_spec)

def _grad_fn(func_graph, grads):
    if False:
        i = 10
        return i + 15
    "The gradient function for each conditional branch.\n\n  This function builds the gradient graph of the corresponding forward-pass\n  conditional branch in `func_graph`. This is done by differentiating\n  func_graph's outputs w.r.t. its inputs.\n\n  Args:\n    func_graph: FuncGraph. The corresponding forward-pass function.\n    grads: The list of input gradient Tensors.\n\n  Returns:\n    The output gradient Tensors.\n  "
    assert len(func_graph.outputs) == len(grads)
    ys = []
    grad_ys = []
    for (y, grad_y) in zip(func_graph.outputs, grads):
        if not backprop_util.IsTrainable(y):
            continue
        ys.append(y)
        grad_ys.append(grad_y)
    result = gradients_util._GradientsHelper(ys, func_graph.inputs, grad_ys=grad_ys, src_graph=func_graph)
    return result

def _create_grad_func(func_graph, grads, name):
    if False:
        for i in range(10):
            print('nop')
    'Returns the FuncGraph representation of _grad_fn.'
    return func_graph_module.func_graph_from_py_func(name, lambda : _grad_fn(func_graph, grads), [], {}, func_graph=_CondGradFuncGraph(name, func_graph))

def _resolve_grad_inputs(cond_graph, grad_graph):
    if False:
        while True:
            i = 10
    'Returns the tensors to pass as inputs to `grad_graph`.\n\n  The `grad_graph` may have external references to\n  1. Its outer graph containing the input gradients. These references are kept\n     as is.\n  2. Tensors in the forward pass graph. These tensors may not be "live"\n     when the gradient is being computed. We replace such references by their\n     corresponding tensor in `cond_graph.outer_graph`. In the case of nested\n     control flow or functions, the gradient logic handling\n     `grad_graph.outer_graph` will make sure the tensor from\n     `cond_graph.outer_graph` is also correctly captured.\n\n  Args:\n    cond_graph: FuncGraph. The forward-pass function.\n    grad_graph: FuncGraph. The gradients function.\n\n  Returns:\n    A list of inputs tensors to be passed to grad_graph.\n  '
    new_inputs = []
    for t in grad_graph.external_captures:
        if t.graph != grad_graph.outer_graph:
            assert t.graph == cond_graph
            for (i, output) in enumerate(t.graph.outputs):
                if output is t:
                    t = t.graph._forward_cond.outputs[i]
                    break
            else:
                for (i, output) in enumerate(t.graph.internal_captures):
                    if output is t:
                        t = t.graph.external_captures[i]
                        break
                else:
                    raise ValueError('Could not find external tensor capture {tensor} in captures or outputs'.format(tensor=t))
            assert t.graph == cond_graph.outer_graph
        new_inputs.append(t)
    return new_inputs

def _get_intermediates(func_graph):
    if False:
        print('Hello World!')
    'Returns intermediate tensors of `func_graph` for gradient computation.'
    intermediates = []
    for op in func_graph.get_operations():
        for t in op.outputs:
            if t in func_graph.inputs:
                continue
            if t in func_graph.outputs:
                continue
            if t.dtype is dtypes.resource:
                continue
            if op.type == 'MutexLock':
                continue
            intermediates.append(t)
    return intermediates

def _make_intermediates_match(branch_graphs, branch_optionals):
    if False:
        i = 10
        return i + 15
    'Returns new optionals lists that have matching signatures.\n\n  This is done by mirroring each list in the other using none optionals.\n  There is no merging of like optionals.\n\n  Args:\n    branch_graphs: `list` of `FuncGraph`.\n    branch_optionals: `list` of `list`s of optional `Tensor`s from other\n      branch_graphs\n\n  Returns:\n    A `list` of `list`s of `Tensor`s for each branch_graph. Each list has the\n    same number of `Tensor`s, all of which will be optionals of the same\n    shape/type.\n  '
    new_branch_optionals = []
    intermediates_size = max((len(o) for o in branch_optionals))
    for (i, branch_graph) in enumerate(branch_graphs):
        other_optionals = _create_none_optionals(branch_graph, intermediates_size - len(branch_optionals[i]))
        new_branch_optionals.append(branch_optionals[i] + other_optionals)
    return new_branch_optionals

def _make_intermediates_match_xla(branch_graphs, branch_intermediates):
    if False:
        print('Hello World!')
    'Like _make_intermediates_match but for the XLA case.'
    new_branch_intermediates = []
    for (i, branch_graph) in enumerate(branch_graphs):
        other_fakeparams = _create_fakeparams(branch_graph, sum((bi for bi in branch_intermediates if bi is not branch_intermediates[i]), []))
        num_preceding = sum((len(bi) for bi in branch_intermediates[:i]))
        new_branch_intermediates.append(other_fakeparams[:num_preceding] + branch_intermediates[i] + other_fakeparams[num_preceding:])
    return new_branch_intermediates

def _make_inputs_match(branch_graphs, branch_inputs):
    if False:
        print('Hello World!')
    "Modifies branch_graphs so they have the same input signature.\n\n  This method reorders and/or adds parameters to each graph in branch_graphs so\n  they have the same input signature, and updates the 'inputs' and 'captured'\n  fields of each graph accordingly. It uses the input tensors from the outer\n  graph to avoid duplicating shared arguments.\n\n  Args:\n    branch_graphs: a `list` of `FuncGraph`\n    branch_inputs: a `list` of `list`s of `Tensor`s in the outer graph. The\n      inputs for the corresponding graph in `branch_graphs`.\n\n  Returns:\n    A new list of Tensors from the outer graph that are the new inputs for each\n    branch_graph. This is a deduped version of `sum(branch_inputs)`.\n  "
    assert len(branch_graphs) == len(branch_inputs)
    added_inputs = set()
    new_inputs = []
    for branch_in in branch_inputs:
        for tensor in branch_in:
            tensor_id = ops.tensor_id(tensor)
            if tensor_id not in added_inputs:
                added_inputs.add(tensor_id)
                new_inputs.append(tensor)
    for (branch_graph, branch_in) in zip(branch_graphs, branch_inputs):
        input_ids = [ops.tensor_id(t) for t in branch_in]
        branch_input_to_param = dict(zip(input_ids, branch_graph.inputs))
        input_list = []
        for in_t in new_inputs:
            param = branch_input_to_param.get(ops.tensor_id(in_t))
            if param is None:
                param = _create_dummy_input(branch_graph, in_t)
            input_list.append(param)
        branch_graph.inputs = input_list
        branch_graph.function_captures.reset_captures(new_inputs, branch_graph.inputs)
    return new_inputs

def _create_zeros_for_none_grads(forward_graphs, grad_graphs):
    if False:
        return 10
    'Creates zeros for None out grads if at least one branch has non-None grad.\n\n  Args:\n    forward_graphs: List of forward FuncGraphs.\n    grad_graphs: List of grad FuncGraphs.\n  '
    assert len(forward_graphs) == len(grad_graphs)
    branch_outputs = [g.structured_outputs for g in grad_graphs]
    num_outputs_per_branch = [len(outs) for outs in branch_outputs]
    assert len(set(num_outputs_per_branch)) == 1, num_outputs_per_branch
    for (output_idx, branch_outs) in enumerate(zip(*branch_outputs)):
        if any((t is None for t in branch_outs)) and any((t is not None for t in branch_outs)):
            for (branch_index, t) in enumerate(branch_outs):
                if t is None:
                    with grad_graphs[branch_index].as_default():
                        zeros = default_gradient.zeros_like(forward_graphs[branch_index].inputs[output_idx])
                        grad_graphs[branch_index].structured_outputs[output_idx] = zeros
    for grad_graph in grad_graphs:
        grad_graph.outputs = [t for t in func_graph_module.flatten(grad_graph.structured_outputs) if t is not None]

def _make_output_composite_tensors_match(op_type, branch_graphs):
    if False:
        return 10
    "Modifies each branch_graph's outputs to have the same output signature.\n\n  Currently the only transformation implemented is turning a Tensor into an\n  equivalent IndexedSlices if the other branch returns an IndexedSlices.\n  Updates branch_graph.{outputs,structured_outputs} for each branch_graph in\n  branch_graphs.\n\n  Args:\n    op_type: _COND or _CASE\n    branch_graphs: `list` of `FuncGraph`\n\n  Raises:\n    TypeError: if a set of outputs cannot be rewritten.\n  "
    assert branch_graphs
    branch_outputs = [g.structured_outputs for g in branch_graphs]
    outputs_per_branch = list((len(outs) for outs in branch_outputs))
    assert len(set(outputs_per_branch)) == 1, outputs_per_branch
    for (output_idx, branch_outs) in enumerate(zip(*branch_outputs)):
        if len(set((type(out) for out in branch_outs))) == 1:
            continue
        if not any((isinstance(out, indexed_slices.IndexedSlices) for out in branch_outs)):
            continue
        for (branch_idx, branch_out) in enumerate(branch_outs):
            if isinstance(branch_out, indexed_slices.IndexedSlices):
                continue
            elif isinstance(branch_out, tensor_lib.Tensor):
                with branch_graphs[branch_idx].as_default():
                    branch_outputs[branch_idx][output_idx] = math_ops._as_indexed_slices(branch_out)
            else:
                raise TypeError('Cannot reconcile {op_name} {output_idx}-th outputs:\n  outputs from all branches: {outputs}'.format(op_name='tf.cond' if op_type == _COND else 'tf.switch_case', output_idx=output_idx, outputs=branch_outs))
    for (branch_graph, branch_outs) in zip(branch_graphs, branch_outputs):
        branch_graph.structured_outputs = branch_outs
        branch_graph.outputs = [t for t in func_graph_module.flatten(branch_outs) if t is not None]

def _make_indexed_slices_indices_types_match(op_type, branch_graphs):
    if False:
        i = 10
        return i + 15
    'Match dtype of IndexedSlices.indices in outputs of branch_graphs.'
    assert branch_graphs
    indexed_slice_indices = []
    current_index = 0
    branch_outputs_flat_with_composites = [nest.flatten(branch_graph.structured_outputs, expand_composites=False) for branch_graph in branch_graphs]
    outs_per_branch = [len(outs) for outs in branch_outputs_flat_with_composites]
    assert len(set(outs_per_branch)) == 1, outs_per_branch
    for (output_idx, branch_outs) in enumerate(zip(*branch_outputs_flat_with_composites)):
        if len(set((isinstance(out, indexed_slices.IndexedSlices) for out in branch_outs))) != 1:
            raise TypeError('Cannot reconcile tf.{op_name} {output_idx}-th outputs:\n  branches returned: {outputs}'.format(op_name='cond' if op_type == _COND else 'switch_case', output_idx=output_idx, outputs=branch_outs))
        if isinstance(branch_outs[0], indexed_slices.IndexedSlices):
            indexed_slice_indices.append(current_index + 1)
        if nest.is_nested_or_composite(branch_outs[0]):
            current_index += len(nest.flatten(branch_outs[0], expand_composites=True))
        elif branch_outs[0] is not None:
            current_index += 1
    if not indexed_slice_indices:
        return
    if current_index != len(branch_graphs[0].outputs):
        raise ValueError('Insufficient elements in branch_graphs[0].outputs.\nExpected: %i\nActual: %i' % (current_index, len(branch_graphs[0].outputs)))
    for index in indexed_slice_indices:
        if any((bg.outputs[index].dtype not in (dtypes.int32, dtypes.int64) for bg in branch_graphs)):
            raise TypeError('Type of IndexedSlices.indices must be int32 or int64. Found: %s' % str([bg.outputs[index].dtype for bg in branch_graphs]))
        if len(set((bg.outputs[index].dtype for bg in branch_graphs))) != 1:
            for branch_graph in branch_graphs:
                if branch_graph.outputs[index].dtype == dtypes.int32:
                    with branch_graph.as_default():
                        branch_graph.outputs[index] = math_ops.cast(branch_graph.outputs[index], dtypes.int64)
    for branch_graph in branch_graphs:
        branch_graph.structured_outputs = _pack_sequence_as(branch_graph.structured_outputs, branch_graph.outputs)

def _pack_sequence_as(structured_outputs, op_outputs):
    if False:
        return 10
    "Packs the outputs of the gradient If/Case op.\n\n  The branch functions may contain None's in the list of `structured_outputs`.\n  `op_outputs` has those outputs missing. So we need to add those Nones to the\n  list of `op_outputs` and then pack it in the same structure as\n  `structured_outputs`.\n\n  Args:\n    structured_outputs: structured_outputs from one of the branch functions.\n    op_outputs: List of output tensors of the op.\n\n  Returns:\n    `op_outputs` packed like `structured_outputs`.\n  "
    outputs_with_nones = []
    counter = 0
    for output in nest.flatten(structured_outputs, expand_composites=True):
        if output is None:
            outputs_with_nones.append(None)
        else:
            outputs_with_nones.append(op_outputs[counter])
            counter += 1
    return func_graph_module.pack_sequence_as(structured_outputs, outputs_with_nones)

def _wrap_intermediates(func_graph, intermediates):
    if False:
        i = 10
        return i + 15
    with func_graph.as_default():
        return [gen_optional_ops.optional_from_value([t]) for t in intermediates]

def _create_dummy_input(func_graph, template_tensor):
    if False:
        return 10
    'Creates tensors in func_graph to represent template_tensors.\n\n  Args:\n    func_graph: FuncGraph.\n    template_tensor: a tensor in the outer graph.\n\n  Returns:\n    A tensor in func_graph.\n  '
    with func_graph.as_default():
        return array_ops.placeholder(template_tensor.dtype, shape=template_tensor.shape)

def _create_none_optionals(func_graph, n):
    if False:
        i = 10
        return i + 15
    'Creates `n` `None` optionals in func_graph.\n\n  Args:\n    func_graph: FuncGraph.\n    n: `int` the number of `None` optionals to make.\n\n  Returns:\n    A list of tensors in func_graph.\n  '
    with func_graph.as_default():
        return [gen_optional_ops.optional_none() for _ in range(n)]

def _convert_dynamic_dimension_to_zero(shape):
    if False:
        return 10
    'Converts dynamic dimensions in `shape` to zero.\n\n  The fake params created to match the intermediates captured in other branches\n  could have dynamic dimensions. But the XLA shape is not able to handle\n  dynamic dimensions in TF TensorShape. Setting the dynamic dimensions to\n  size zero will help avoid failing safety checks in bridge. When XLA\n  DynamicConditional op reconciles branch differences, XLA will replace the\n  dimension size 0 with a bounded dimension determined from the shape of\n  real argument in the other branch.\n\n  Note: Rank unknown shapes are returned as they are.\n\n  Args:\n    shape: The TensorShape of fake param.\n\n  Returns:\n    The new TensorShape with dynamic dimensions set to zero.\n  '
    if shape.rank is None:
        return shape
    return tensor_shape.TensorShape([0 if d is None else d for d in shape.as_list()])

def _create_fakeparams(func_graph, template_tensors):
    if False:
        while True:
            i = 10
    'Creates FakeParams for the XLA case.'
    with func_graph.as_default():
        return [gen_functional_ops.fake_param(dtype=t.dtype, shape=_convert_dynamic_dimension_to_zero(t.shape)) for t in template_tensors]

def _check_same_outputs(op_type, graphs):
    if False:
        i = 10
        return i + 15
    'Raises an error if `graphs` have different outputs.'

    def error(branch_idx, error_detail):
        if False:
            print('Hello World!')
        raise TypeError('{b0_name} and {bn_name} arguments to {op_name} must have the same number, type, and overall structure of return values.\n\n{b0_name} output: {b0_out}\n{bn_name} output: {bn_out}\n\nError details:\n{detail}'.format(b0_name='true_fn' if op_type == _COND else 'branches[0]', bn_name='false_fn' if op_type == _COND else 'branches[{}]'.format(branch_idx), op_name='tf.cond' if op_type == _COND else 'tf.switch_case', b0_out=graphs[0].structured_outputs, bn_out=graphs[branch_idx].structured_outputs, detail=error_detail))
    for b in range(1, len(graphs)):
        try:
            nest.assert_same_structure(graphs[0].structured_outputs, graphs[b].structured_outputs, expand_composites=True)
        except (ValueError, TypeError) as e:
            error(b, str(e))
        op_type_str = 'cond' if op_type == _COND else 'case'
        if len(graphs[0].outputs) != len(graphs[b].outputs):
            raise ValueError('Lengths of branch outputs of {op_type} must match.\nlen(graphs[0].outputs): {len_0}\nlen(graphs[{b}].outputs): {len_b}\n'.format(op_type=op_type_str, len_0=len(graphs[0].outputs), b=b, len_b=len(graphs[b].outputs)))
        for (b0_out, bn_out) in zip(graphs[0].outputs, graphs[b].outputs):
            if b0_out.dtype != bn_out.dtype:
                error(b, '%s and %s have different types' % (b0_out, bn_out))

def _get_output_shapes(*branch_graph_outputs):
    if False:
        i = 10
        return i + 15
    output_shapes = []
    for out_by_branch in zip(*branch_graph_outputs):
        shape = out_by_branch[0].shape
        for other_out in out_by_branch[1:]:
            shape = shape.most_specific_compatible_shape(other_out.shape)
        output_shapes.append(shape)
    return output_shapes

def _copy_handle_data(external_tensors, *branch_graph_outputs):
    if False:
        i = 10
        return i + 15
    'Combines shapes in handle data and sets metadata on `external_tensors`.'
    for tensors in zip(external_tensors, *branch_graph_outputs):
        external = tensors[0]
        internal = tensors[1:]
        internal_handle_data = []
        for tensor in internal:
            handle_data = handle_data_util.get_resource_handle_data(tensor)
            if not handle_data.is_set or len(handle_data.shape_and_type) != 1:
                break
            internal_handle_data.append(handle_data)
        else:
            combined_shape = tensor_shape.TensorShape(None)
            combined_dtype = None
            for handle_data in internal_handle_data:
                handle_shape = tensor_shape.TensorShape(handle_data.shape_and_type[0].shape)
                combined_shape = combined_shape.most_specific_compatible_shape(handle_shape)
                if combined_dtype is None:
                    combined_dtype = handle_data.shape_and_type[0].dtype
                elif handle_data.shape_and_type[0].dtype != combined_dtype:
                    combined_dtype = types_pb2.DT_INVALID
            combined_handle_data = internal_handle_data[0]
            combined_handle_data.shape_and_type[0].shape.CopyFrom(combined_shape.as_proto())
            combined_handle_data.shape_and_type[0].dtype = combined_dtype
            handle_data_util.set_handle_data(external, combined_handle_data)

def verify_captures(op_type, branch_graphs):
    if False:
        for i in range(10):
            print('nop')
    "Verify that a branch's tensor is not accessed in another branch fn."
    other_branch_graphs = {g: i for (i, g) in enumerate(branch_graphs)}
    for (i, branch_graph) in enumerate(branch_graphs):
        for t in branch_graph.external_captures:
            if not isinstance(t, ops.EagerTensor) and t.graph in other_branch_graphs:
                branch_names = ['true_fn', 'false_fn'] if op_type == _COND else ['branch {}'.format(bi) for bi in range(len(branch_graphs))]
                raise ValueError('Tensor {tname} in {b0name} is accessed from {b1name}.'.format(tname=t.name, b0name=branch_names[other_branch_graphs[t.graph]], b1name=branch_names[i]))

class _CondGradFuncGraph(util.CondBranchFuncGraph):
    """FuncGraph for the gradient function of the branch of an If op.

  Handles wrapping and unwrapping intermediate values that are captured by the
  gradient computation in optionals.

  Attributes:
    op_needs_rewrite: True if any intermediates were captured, meaning the
      forward If op needs to be written to output the wrapped intermediates.
  """

    def __init__(self, name, forward_graph):
        if False:
            i = 10
            return i + 15
        super(_CondGradFuncGraph, self).__init__(name, collections=ops.get_default_graph()._collections)
        self.op_needs_rewrite = False
        self._forward_graph = forward_graph
        self._indirect_captures = {}
        self._wrapped_intermediates = collections.OrderedDict()
        self._xla_intermediates = []
        self._captured_constants = {}

    @property
    def wrapped_intermediates(self):
        if False:
            while True:
                i = 10
        'The optional-wrapped intermediates captured from the forward graph.'
        return list(self._wrapped_intermediates.values())

    @property
    def xla_intermediates(self):
        if False:
            i = 10
            return i + 15
        'Raw intermediates captured from the forward graph if XLA is enabled.'
        return self._xla_intermediates

    def _capture_helper(self, tensor, name):
        if False:
            i = 10
            return i + 15
        if tensor.graph is not self._forward_graph or any((tensor is t for t in self._forward_graph.inputs)) or any((tensor is t for t in self._forward_graph.outputs)):
            return super(_CondGradFuncGraph, self)._capture_helper(tensor, name)
        tensor_id = ops.tensor_id(tensor)
        if tensor_id in self._captured_constants:
            return self._captured_constants[tensor_id]
        elif constant_op.is_constant(tensor):
            self._captured_constants[tensor_id] = constant_op.constant(tensor_util.constant_value(tensor), dtype=tensor.dtype)
            return self._captured_constants[tensor_id]
        if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
            if all((tensor is not capture for capture in self.external_captures)):
                self.xla_intermediates.append(tensor)
                self.op_needs_rewrite = True
            return super(_CondGradFuncGraph, self)._capture_helper(tensor, name)
        captured_tensor = self._indirect_captures.get(tensor_id)
        if captured_tensor is not None:
            return captured_tensor
        if tensor.dtype == dtypes.resource:
            index = util.resource_input_index(tensor.name, [t.name for t in self._forward_graph.inputs], {op.name: op.node_def for op in self._forward_graph.get_operations()}, self._forward_graph._functions)
            captured_tensor = super(_CondGradFuncGraph, self)._capture_helper(self._forward_graph.inputs[index], name)
        else:
            if tensor_id not in self._wrapped_intermediates:
                for consumer in tensor.consumers():
                    if consumer.type == 'OptionalFromValue' and any((consumer.outputs[0] is output for output in self._forward_graph.outputs)):
                        optional = consumer.outputs[0]
                        break
                else:
                    with self._forward_graph.as_default():
                        optional = gen_optional_ops.optional_from_value([tensor])
                    self.op_needs_rewrite = True
                self._wrapped_intermediates[tensor_id] = optional
            optional = self._wrapped_intermediates[tensor_id]
            captured_optional = super(_CondGradFuncGraph, self)._capture_helper(optional, name)
            captured_tensor = gen_optional_ops.optional_get_value(captured_optional, [tensor.dtype], [tensor.shape])[0]
        self._indirect_captures[tensor_id] = captured_tensor
        return captured_tensor

def indexed_case(branch_index, branch_fns, name='indexed_case', lower_using_switch_merge=None):
    if False:
        print('Hello World!')
    'Like conv_v2, except emits a Case op instead of an If.'
    if isinstance(branch_index, int):
        raise TypeError('branch_index must not be a Python int', branch_index)
    with ops.name_scope(name) as scope:
        branch_names = [util.unique_fn_name(scope, 'branch{}'.format(b)) for b in range(len(branch_fns))]
        add_control_dependencies = ops.get_default_graph()._add_control_dependencies
        branch_index = ops.convert_to_tensor(branch_index, name='branch_index')
        branch_graphs = []
        for (branch_name, branch_fn) in zip(branch_names, branch_fns):
            branch_graphs.append(func_graph_module.func_graph_from_py_func(branch_name, branch_fn, [], {}, func_graph=util.CondBranchFuncGraph(branch_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies, op_return_value=branch_index))
        verify_captures(_CASE, branch_graphs)
        return _build_case(branch_index, branch_graphs, [g.external_captures for g in branch_graphs], name=scope, lower_using_switch_merge=lower_using_switch_merge)

@ops.RegisterGradient('Case')
@ops.RegisterGradient('StatelessCase')
def _CaseGrad(op, *grads):
    if False:
        while True:
            i = 10
    'The gradient of a Case op produced by tf.switch_case.'
    case_op = op.outputs[0].op
    branch_graphs = get_func_graphs(case_op)
    assert branch_graphs
    for branch_graph in branch_graphs:
        assert branch_graph.outer_graph == case_op.graph
    branch_grad_graphs = []
    for branch_graph in branch_graphs:
        branch_grad_graphs.append(_create_grad_func(branch_graph, grads, util.unique_grad_fn_name(branch_graph.name)))
    _create_zeros_for_none_grads(branch_graphs, branch_grad_graphs)
    if any((g.op_needs_rewrite for g in branch_grad_graphs)):
        if control_flow_util.GraphOrParentsInXlaContext(ops.get_default_graph()):
            branches_intermediates = [branch_grad_graph.xla_intermediates for branch_grad_graph in branch_grad_graphs]
            extra_branch_outputs = _make_intermediates_match_xla(branch_graphs, branches_intermediates)
        else:
            branch_intermediates = [g.wrapped_intermediates for g in branch_grad_graphs]
            extra_branch_outputs = _make_intermediates_match(branch_graphs, branch_intermediates)
        for (branch_graph, extra_outputs) in zip(branch_graphs, extra_branch_outputs):
            branch_graph.outputs.extend(extra_outputs)
        _check_same_outputs(_CASE, branch_graphs)
        for branch_graph in branch_graphs:
            branch_graph.name += '_rewritten'
        case_op._set_func_list_attr('branches', [util.create_new_tf_function(branch_graph) for branch_graph in branch_graphs])
        case_op._set_type_list_attr('Tout', branch_graphs[0].output_types)
        case_op._set_shape_list_attr('output_shapes', branch_graphs[0].output_shapes)
        case_op._add_outputs([t.dtype for t in extra_branch_outputs[0]], [t.shape for t in extra_branch_outputs[0]])
    branches_grad_inputs = [_resolve_grad_inputs(branch_graph, branch_grad_graph) for (branch_graph, branch_grad_graph) in zip(branch_graphs, branch_grad_graphs)]
    _make_output_composite_tensors_match(_CASE, branch_grad_graphs)
    try:
        lowering = case_op._get_attr_bool('_lower_using_switch_merge')
    except errors_impl.NotFoundError:
        lowering = None
    outputs = _build_case(case_op.inputs[0], branch_grad_graphs, branches_grad_inputs, name='gradient', lower_using_switch_merge=lowering)
    return [None] + outputs

def _build_case(branch_index, branch_graphs, branch_inputs, name=None, lower_using_switch_merge=None):
    if False:
        print('Hello World!')
    "Creates an `Case` op from `branch_index`, branch graphs and inputs.\n\n  Note that this modifies `branch_graphs` to make the inputs match, and to\n  output all intermediates values so they're available for the gradient\n  computation.\n\n  `branch_graphs` need not have the same input types, but they must\n  have the same output types.\n\n  Args:\n    branch_index: integer Tensor\n    branch_graphs: List of FuncGraph\n    branch_inputs: List of lists of Tensors to be passed to corresponding\n      branch_graph as input.\n    name: the name for the Case op.\n    lower_using_switch_merge: Lower this op using switch merge ops (optional).\n\n  Returns:\n    A list of Tensors which are the outputs of the Case op. Does not include\n    added intermediate outputs.\n  "
    _make_indexed_slices_indices_types_match(_CASE, branch_graphs)
    _check_same_outputs(_CASE, branch_graphs)
    case_inputs = _make_inputs_match(branch_graphs, branch_inputs)
    stateful_ops = []
    for bg in branch_graphs:
        stateful_ops.extend([op for op in bg.get_operations() if auto_control_deps.op_is_stateful(op)])
    if stateful_ops:
        op_fn = gen_functional_ops.case
    else:
        op_fn = gen_functional_ops.stateless_case
    with ops.control_dependencies(sum((list(bg.function_captures.control) for bg in branch_graphs), [])):

        def _make_op(inputs):
            if False:
                return 10
            (case_op, tensors) = util.get_op_and_outputs(op_fn(branch_index, inputs, [t.dtype for t in branch_graphs[0].outputs], [util.create_new_tf_function(g) for g in branch_graphs], output_shapes=_get_output_shapes(*[g.outputs for g in branch_graphs]), name=name))
            _copy_handle_data(tensors, *[g.outputs for g in branch_graphs])
            if case_op is not None:
                util.maybe_set_lowering_attr(case_op, lower_using_switch_merge)
                util.maybe_propagate_compile_time_consts_in_xla(case_op)
                _set_read_only_resource_inputs_attr(case_op, branch_graphs)
                case_op.graph.prevent_fetching(case_op)
                for (i, bg) in enumerate(branch_graphs):
                    bg.outer_graph = ops.get_default_graph()
                    setattr(case_op, '_branch_graph_{}'.format(i), bg)
            return tensors
        tensors = util.run_as_function_for_tape_gradients(_make_op, case_inputs)
    tensors = [array_ops.identity(t) for t in tensors]
    return _pack_sequence_as(branch_graphs[0].structured_outputs, tensors)

def _set_read_only_resource_inputs_attr(op, branch_graphs):
    if False:
        return 10
    'Sets the list of resource inputs which are read-only.\n\n  This is used by AutomaticControlDependencies.\n\n  Args:\n    op: If or Case Operation.\n    branch_graphs: List of branch FuncGraphs.\n  '
    read_only_indices = set(range(len(op.inputs) - 1))
    for branch_graph in branch_graphs:
        assert len(branch_graph.inputs) == len(op.inputs) - 1, 'should never happen'
        if not read_only_indices:
            break
        branch_read_only_indices = acd.get_read_only_resource_input_indices_graph(branch_graph)
        read_only_indices = read_only_indices.intersection(branch_read_only_indices)
    read_only_indices = [i + 1 for i in read_only_indices]
    ops.set_int_list_attr(op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR, sorted(read_only_indices))