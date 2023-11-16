"""while_v2 and gradient.

This is a version of while_loop that emits a single While op, as well as the
gradient function for While ops produced by while_loop. This will eventually
replace the current tf.while_loop implementation once it reaches feature and
performance parity.
"""
import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.eager import backprop_util
from tensorflow.python.framework import auto_control_deps_utils as acd
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_spec
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import control_flow_util as util_v1
from tensorflow.python.ops import control_flow_util_v2 as util
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import list_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import while_v2_indexed_slices_rewriter
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity
from tensorflow.python.util import variable_utils

def while_loop(cond, body, loop_vars, shape_invariants=None, parallel_iterations=10, maximum_iterations=None, name=None, return_same_structure=True, back_prop=True):
    if False:
        for i in range(10):
            print('nop')
    'Like tf.while_loop, except emits a single While op.'
    loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)
    orig_loop_vars = loop_vars
    flat_orig_loop_vars = nest.flatten(orig_loop_vars, expand_composites=True)
    len_orig_loop_vars = len(orig_loop_vars)
    loop_vars = _tensor_array_to_flow(loop_vars)
    loop_vars = nest.map_structure(indexed_slices.internal_convert_to_tensor_or_indexed_slices, loop_vars, expand_composites=True)
    if shape_invariants is not None:
        loop_vars_signature = nest.map_structure(control_flow_ops._shape_invariant_to_type_spec, loop_vars, shape_invariants)
    else:
        loop_vars_signature = nest.map_structure(control_flow_ops._shape_invariant_to_type_spec, loop_vars)
    flat_shape_invariants = nest.map_structure(lambda spec: spec.shape, nest.flatten(loop_vars_signature, expand_composites=True))
    if not name:
        name = 'while'
    with ops.name_scope(name) as scope:
        with ops.name_scope(None):
            cond_name = util.unique_fn_name(scope, 'cond')
            body_name = util.unique_fn_name(scope, 'body')
        maximum_iterations_loop_var = _build_maximum_iterations_loop_var(maximum_iterations)
        loop_counter = constant_op.constant(0, dtype=maximum_iterations_loop_var.dtype if maximum_iterations is not None else None, name='loop_counter')
        loop_vars = [loop_counter, maximum_iterations_loop_var] + list(loop_vars)
        func_graph_signature = [tensor_spec.TensorSpec.from_tensor(loop_counter), tensor_spec.TensorSpec.from_tensor(maximum_iterations_loop_var)] + list(loop_vars_signature)
        add_control_dependencies = ops.get_default_graph()._add_control_dependencies

        def wrapped_cond(loop_counter, maximum_iterations_arg, *args):
            if False:
                while True:
                    i = 10
            'Extra `cond` wrapper that can handle the extra counter loop_var.'
            pred = cond(*_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args))
            if tensor_util.is_tf_type(pred) and (pred.shape.dims is None or pred.shape.dims):
                pred = array_ops.squeeze_v2(pred)
            if maximum_iterations is None:
                return pred
            else:
                return math_ops.logical_and(loop_counter < maximum_iterations_arg, pred)
        cond_graph = func_graph_module.func_graph_from_py_func(cond_name, wrapped_cond, [], {}, signature=func_graph_signature, func_graph=util.WhileCondFuncGraph(cond_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies)

        def wrapped_body(loop_counter, maximum_iterations_arg, *args):
            if False:
                print('Hello World!')
            'Loop body augmented with counter update.\n\n      Args:\n        loop_counter: Loop counter which needs to be incremented in the body.\n        maximum_iterations_arg: Maximum iterations of the loop.\n        *args: List of args\n\n      Returns:\n        A list of tensors the same length as args.\n      '
            _copy_handle_data(nest.flatten(loop_vars[2:], expand_composites=True), nest.flatten(args, expand_composites=True))
            for t in cond_graph.external_captures:
                ops.get_default_graph().capture(t)
            outputs = body(*_pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, args))
            if not nest.is_nested(outputs):
                outputs = [outputs]
            try:
                nest.assert_same_structure(outputs, orig_loop_vars, check_types=False, expand_composites=True)
            except ValueError:
                vars1 = variable_utils.convert_variables_to_tensors(outputs)
                vars2 = variable_utils.convert_variables_to_tensors(orig_loop_vars)
                nest.assert_same_structure(vars1, vars2, check_types=False, expand_composites=True)
            outputs = _tensor_array_to_flow(outputs)
            return [loop_counter + 1, maximum_iterations_arg] + list(outputs)
        body_graph = func_graph_module.func_graph_from_py_func(body_name, wrapped_body, [], {}, signature=func_graph_signature, func_graph=util.WhileBodyFuncGraph(body_name, collections=ops.get_default_graph()._collections), add_control_dependencies=add_control_dependencies)
        deferred_external_captures = nest.flatten([c() for c in body_graph.deferred_external_captures], expand_composites=True)
        loop_vars = loop_vars + body_graph.external_captures + deferred_external_captures
        body_graph.outputs.extend(body_graph.internal_captures)
        body_graph.outputs.extend(body_graph.deferred_internal_captures)
        with cond_graph.as_default():
            num_cond_captures = len(cond_graph.external_captures)
            assert cond_graph.external_captures == body_graph.external_captures[:num_cond_captures]
            _duplicate_body_captures_in_cond(cond_graph, body_graph.external_captures[num_cond_captures:] + deferred_external_captures)
        num_flattened_outputs = len(nest.flatten(orig_loop_vars, expand_composites=True))
        first_loop_var_index = 2
        _check_shapes_compat(body_graph.outputs[first_loop_var_index:first_loop_var_index + num_flattened_outputs], flat_shape_invariants, nest.flatten(loop_vars[first_loop_var_index:first_loop_var_index + len_orig_loop_vars], expand_composites=True))
        num_original_outputs = len(body_graph.outputs)
        if back_prop and util.output_all_intermediates():
            intermediate_tensors = _get_intermediates(body_graph)
            for intermediate_tensor in intermediate_tensors:
                tensor_list = list_ops.empty_tensor_list(element_dtype=intermediate_tensor.dtype, element_shape=intermediate_tensor.shape, max_num_elements=maximum_iterations)
                loop_vars.append(tensor_list)
                with cond_graph.as_default():
                    cond_graph.capture(tensor_list)
                with body_graph.as_default():
                    appended_tensor_list = list_ops.tensor_list_push_back(tensor_list, intermediate_tensor)
                    body_graph.outputs.append(appended_tensor_list)
        flattened_loop_vars = nest.flatten(loop_vars, expand_composites=True)
        _check_num_inputs_outputs(cond_graph, body_graph, len(flattened_loop_vars))
        _check_inputs_outputs_types_match(body_graph, flattened_loop_vars)
        with ops.control_dependencies(list(cond_graph.function_captures.control) + list(body_graph.function_captures.control)):
            output_shapes = [t.shape for t in body_graph.outputs]
            orig_loop_vars_range = slice(first_loop_var_index, first_loop_var_index + num_flattened_outputs)
            output_shapes[orig_loop_vars_range] = flat_shape_invariants
            outputs = _build_while_op(flattened_loop_vars, cond_graph, body_graph, output_shapes=output_shapes, parallel_iterations=parallel_iterations, name=scope, num_original_outputs=num_original_outputs)
        if not ops.get_default_graph().building_function:
            outputs = tuple((array_ops.identity(t) for t in outputs))
    output_loop_vars = outputs[first_loop_var_index:first_loop_var_index + num_flattened_outputs]
    if not back_prop:
        output_loop_vars = [array_ops.stop_gradient(t) for t in output_loop_vars]
    outputs = _pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, output_loop_vars)
    if return_same_structure:
        return outputs
    flattened_outputs = nest.flatten(outputs, expand_composites=True)
    if len(flattened_outputs) == 1:
        return flattened_outputs[0]
    else:
        return outputs

@ops.RegisterGradient('StatelessWhile')
@ops.RegisterGradient('While')
def _WhileGrad(op: ops.Operation, *grads):
    if False:
        i = 10
        return i + 15
    'The gradient of a While op produced by while_loop.'
    while_op = op.outputs[0].op
    cond_graph = _get_graph(while_op, 'cond', '_cond_graph')
    body_graph = _get_graph(while_op, 'body', '_body_graph')
    orig_num_params = len(body_graph.outputs)
    maximum_iterations = op.inputs[1]
    parallel_iterations = op.get_attr('parallel_iterations')
    try:
        num_original_outputs = while_op.get_attr('_num_original_outputs')
    except:
        num_original_outputs = len(while_op.outputs)
    num_intermediates = len(while_op.outputs) - num_original_outputs
    grads = [_preprocess_grad(grad, body_out, while_in, while_out) for (grad, body_out, while_in, while_out) in zip(grads[:num_original_outputs], body_graph.outputs[:num_original_outputs], while_op.inputs[:num_original_outputs], while_op.outputs[:num_original_outputs])] + [None] * num_intermediates
    if getattr(op, 'skip_input_indices', None) is not None:
        captures_start_index = len(body_graph.inputs) - len(body_graph.internal_captures)
        for i in op.skip_input_indices:
            if i >= captures_start_index:
                grads[i] = None
    (ys, xs, non_none_grads) = zip(*[(y, x, grad) for (y, x, grad) in zip(body_graph.outputs, body_graph.inputs, grads) if grad is not None])
    (body_grad_graph, args) = _create_grad_func(ys, xs, non_none_grads, cond_graph, body_graph, util.unique_grad_fn_name(body_graph.name), op, maximum_iterations)
    if body_grad_graph.while_op_needs_rewrite:
        cond_graph.name += '_rewritten'
        body_graph.name += '_rewritten'
        new_inputs = body_grad_graph.extra_inputs
        new_outputs = body_graph.outputs[orig_num_params:]
        while_op._set_func_attr('cond', util.create_new_tf_function(cond_graph))
        while_op._set_func_attr('body', util.create_new_tf_function(body_graph))
        if len(body_graph.output_types) != len(while_op.inputs) + len(new_inputs):
            raise AssertionError(f"Inputs and outputs constructed for the forward op of a While gradient don't match with 'output_types' at  {len(body_graph.output_types)},'inputs' at length {len(while_op.inputs)}, and 'new_inputs' at length {len(new_inputs)}. This doesn't make sense, please file a bug.")
        while_op._set_type_list_attr('T', body_graph.output_types)
        while_op._set_shape_list_attr('output_shapes', body_graph.output_shapes)
        while_op._add_while_inputs(new_inputs)
        while_op._add_outputs([t.dtype for t in new_outputs], [t.shape for t in new_outputs])
        _copy_handle_data(new_outputs, while_op.outputs[orig_num_params:])
    while_op._set_attr('_num_original_outputs', attr_value_pb2.AttrValue(i=len(while_op.outputs)))
    captured_inputs = _resolve_grad_captures(body_graph, body_grad_graph, while_op)
    loop_vars = args + captured_inputs
    loop_vars = while_v2_indexed_slices_rewriter.rewrite_grad_indexed_slices(grads, body_grad_graph, loop_vars, while_op.inputs)

    def grad_cond(counter, unused_maximum_iterations_arg, forward_loop_iters, *unused_args):
        if False:
            for i in range(10):
                print('nop')
        return counter < forward_loop_iters
    grad_cond_name = util.unique_grad_fn_name(op.get_attr('cond').name)
    cond_grad_graph = func_graph_module.func_graph_from_py_func(grad_cond_name, grad_cond, loop_vars, {}, func_graph=util.WhileCondFuncGraph(grad_cond_name))
    _check_num_inputs_outputs(cond_grad_graph, body_grad_graph, len(loop_vars))
    outputs = _build_while_op(loop_vars, cond_grad_graph, body_grad_graph, output_shapes=[t.shape for t in body_grad_graph.outputs], parallel_iterations=parallel_iterations, name='%s_grad' % while_op.name, num_original_outputs=len(body_grad_graph.outputs))
    outputs = [array_ops.identity(t) for t in outputs]
    return _get_structured_grad_output(outputs, grads, body_grad_graph)

def _build_while_op(loop_vars, cond_graph, body_graph, output_shapes, parallel_iterations, name, num_original_outputs):
    if False:
        i = 10
        return i + 15
    'Builds the functional StatelessWhile/While op.'
    cond_stateful_ops = [op for op in cond_graph.get_operations() if op._is_stateful]
    body_stateful_ops = [op for op in body_graph.get_operations() if op._is_stateful]
    if cond_stateful_ops or body_stateful_ops:
        op_fn = gen_functional_ops._while
    else:
        op_fn = gen_functional_ops.stateless_while

    def _make_op(inputs):
        if False:
            while True:
                i = 10
        (while_op, tensors) = util.get_op_and_outputs(op_fn(inputs, util.create_new_tf_function(cond_graph), util.create_new_tf_function(body_graph), output_shapes=output_shapes, parallel_iterations=parallel_iterations, name=name))
        _copy_handle_data(body_graph.outputs, tensors)
        util.maybe_set_lowering_attr(while_op)
        util.maybe_propagate_compile_time_consts_in_xla(while_op)
        _set_read_only_resource_inputs_attr(while_op, [cond_graph, body_graph])
        while_op._set_attr('_num_original_outputs', attr_value_pb2.AttrValue(i=num_original_outputs))
        cond_graph.outer_graph = ops.get_default_graph()
        body_graph.outer_graph = ops.get_default_graph()
        while_op._cond_graph = cond_graph
        while_op._body_graph = body_graph
        return tensors
    return util.run_as_function_for_tape_gradients(_make_op, loop_vars)

def _get_intermediates(func_graph):
    if False:
        i = 10
        return i + 15
    'Returns all tensors in `func_graph` that should be accumulated.'
    intermediates = []
    reverse_captures = dict(((v.ref(), k) for (k, v) in func_graph.captures))
    for op in func_graph.get_operations():
        if op.type == 'Identity':
            continue
        if op.type == 'MutexLock':
            continue
        for o in op.outputs:
            if o is not func_graph.inputs[0] and o.dtype != dtypes.resource and (_get_accumulator(o) is None) and (o.ref() not in reverse_captures):
                intermediates.append(o)
    return intermediates

def _preprocess_grad(grad, body_graph_output, while_op_input, while_op_output):
    if False:
        i = 10
        return i + 15
    'Returns the initial gradient to be used for a given output tensor.\n\n  Args:\n    grad: the original gradient Tensor passed to the gradient function.\n    body_graph_output: the corresponding Tensor in the body graph.\n    while_op_input: the corresponding Tensor input of the While op.\n    while_op_output: the corresponding Tensor output of the While op.\n\n  Returns:\n    A Tensor or None.\n  '
    if not _is_trainable(body_graph_output):
        return None
    if while_op_output.dtype in (dtypes.resource, dtypes.variant) and default_gradient.supports_default_grad(while_op_input) and (grad is None):
        return _zeros_like(while_op_input, while_op_output)
    if isinstance(grad, indexed_slices.IndexedSlices):
        return ops.convert_to_tensor(grad)
    return grad

def _zeros_like(op_input, op_output):
    if False:
        for i in range(10):
            print('nop')
    'Like array_ops.zeros_like() but also accepts resource var handles.'
    if op_output.dtype == dtypes.resource:
        return array_ops.zeros(gen_resource_variable_ops.variable_shape(op_output), dtype=default_gradient.get_zeros_dtype(op_input))
    return array_ops.zeros_like(op_output)

def _is_trainable(tensor):
    if False:
        for i in range(10):
            print('nop')
    'Returns whether the given tensor is trainable.'
    if not backprop_util.IsTrainable(tensor):
        return False
    if tensor.op.type == 'TensorListPopBack' and tensor.value_index == 0:
        assert tensor.dtype == dtypes.variant
        element_type = tensor.op.get_attr('element_dtype')
        return backprop_util.IsTrainable(element_type)
    return True

def _get_graph(while_op, func_attr_name, attr_graph_name):
    if False:
        i = 10
        return i + 15
    'Returns `FuncGraph` for the given function attribute.\n\n  Args:\n    while_op: The While Operation.\n    func_attr_name: string\n    attr_graph_name: cached forward graph name\n\n  Returns:\n    `FuncGraph`\n  '
    func_graph = getattr(while_op, attr_graph_name, None)
    if func_graph is None:
        input_shapes = [tensor_shape.TensorShape(s) for s in while_op.get_attr('output_shapes')]
        func_name = while_op.get_attr(func_attr_name).name
        func_graph = util.get_func_graph(while_op, input_shapes, func_name)
    func_graph._while = while_op
    return func_graph

def _create_grad_func(ys, xs, grads, cond_graph, body_graph, name, while_op, maximum_iterations):
    if False:
        for i in range(10):
            print('nop')
    'Builds and returns the gradient FuncGraph of `func_graph` and its args.\n\n  The returned grad_func_graph must be called with the returned\n  args + grad_func_graph.captures.\n\n  Args:\n    ys: A `Tensor` or list of tensors to be differentiated.\n    xs: A `Tensor` or list of tensors to be used for differentiation.\n    grads: The incoming grads for `ys`.\n    cond_graph: FuncGraph for the forward cond function.\n    body_graph: FuncGraph for the forward body function.\n    name: Name of the returned gradient function.\n    while_op: The forward While op.\n    maximum_iterations: Tensor. The maximum number of iterations.\n\n  Returns:\n    2-tuple of (grad_func_graph, args).\n  '
    assert len(ys) == len(grads)
    total_iters = while_op.outputs[0]
    counter = constant_op.constant(0, dtype=total_iters.dtype, name='grad_counter')
    body_graph_inputs = object_identity.ObjectIdentitySet(body_graph.inputs)
    body_graph_outputs = object_identity.ObjectIdentitySet(body_graph.outputs)
    args = [counter, maximum_iterations, total_iters] + list(grads)
    grad_func_graph = func_graph_module.func_graph_from_py_func(name, lambda *args: _grad_fn(ys, xs, args, body_graph), args, {}, func_graph=_WhileBodyGradFuncGraph(name, cond_graph, body_graph, maximum_iterations, while_op, body_graph_inputs, body_graph_outputs))
    for (external_capture, internal_capture) in grad_func_graph.captures:
        if ops.tensor_id(internal_capture) in grad_func_graph.internal_capture_to_output:
            new_output = grad_func_graph.internal_capture_to_output[ops.tensor_id(internal_capture)]
        else:
            raise ValueError(f'Tensor {str(internal_capture)} which captures {str(external_capture)} is in list of internal_captures but not in internal_capture_to_output.')
        grad_func_graph.outputs.append(new_output)
        grad_func_graph.structured_outputs.append(new_output)
    return (grad_func_graph, args)

def _grad_fn(ys, xs, args, func_graph):
    if False:
        for i in range(10):
            print('nop')
    "Computes the gradient of `func_graph` in the current graph.\n\n  This function builds the gradient graph of the corresponding forward-pass\n  `func_graph` by differentiating `func_graph`'s outputs w.r.t. its inputs.\n\n  Args:\n    ys: A `Tensor` or list of tensors to be differentiated.\n    xs: A `Tensor` or list of tensors to be used for differentiation.\n    args: The input arguments.\n      args[0] - Loop counter\n      args[1] - Total number of iterations.\n      args[2] - maximum_iterations.\n      args[3:] - Incoming gradients for `ys`.\n    func_graph: function.FuncGraph. The corresponding forward-pass function.\n\n  Returns:\n    The output gradient Tensors.\n  "
    grad_ys = args[3:]
    grad_outs = gradients_util._GradientsHelper(ys, xs, grad_ys=grad_ys, src_graph=func_graph, unconnected_gradients='zero')
    assert all((g is not None for g in grad_outs))
    counter = args[0]
    maximum_iterations = args[1]
    total_iters = args[2]
    return [counter + 1, maximum_iterations, total_iters] + grad_outs

def _resolve_grad_captures(body_graph, body_grad_graph, while_op):
    if False:
        i = 10
        return i + 15
    'Returns the tensors to pass as captured inputs to `body_grad_graph`.\n\n  `body_grad_graph` may have external references to:\n  1. Its outer graph containing the input gradients. These are left as-is.\n  2. Accumulators captured from the forward-pass graph. These should have been\n     added as `while_op` outputs after the gradient graph was built. We replace\n     these with the corresponding output of `while_op`, i.e. a tensor in\n     `body_graph.outer_graph`. In the case of nested control flow or functions,\n     the gradient logic handling `body_grad_graph.outer_graph` will make sure\n     the tensor from `body_graph.outer_graph` is also correctly captured.\n\n  Args:\n    body_graph: FuncGraph. The forward-pass body function.\n    body_grad_graph: FuncGraph. The body gradients function.\n    while_op: The forward-pass While Operation calling `body_graph`.\n\n  Returns:\n    A list of input tensors to be passed as the captured inputs to\n    `body_grad_graph`.\n  '
    new_capture_inputs = []
    for t in body_grad_graph.external_captures:
        if t.graph == body_graph:
            for (i, output) in enumerate(t.graph.outputs):
                if output is t:
                    t = while_op.outputs[i]
                    break
            assert t.graph == body_graph.outer_graph
        new_capture_inputs.append(t)
    return new_capture_inputs

def _get_structured_grad_output(outputs, grads, body_grad_graph):
    if False:
        i = 10
        return i + 15
    'Returns the values that should be returned from the while grad function.\n\n  Args:\n    outputs: the raw Tensor outputs of the grad While op.\n    grads: the input gradients to the gradient function.\n    body_grad_graph: _WhileBodyGradFuncGraph.\n\n  Returns:\n    A list of gradient values. May include Nones.\n  '
    result = []
    outputs_idx = 3
    structured_outputs_idx = 3
    for g in grads:
        if g is None:
            result.append(None)
            continue
        output = body_grad_graph.structured_outputs[structured_outputs_idx]
        structured_outputs_idx += 1
        if isinstance(output, indexed_slices.IndexedSlices):
            result.append(indexed_slices.IndexedSlices(values=outputs[outputs_idx], indices=outputs[outputs_idx + 1], dense_shape=outputs[outputs_idx + 2]))
            outputs_idx += 3
        else:
            assert isinstance(output, tensor_lib.Tensor)
            result.append(outputs[outputs_idx])
            outputs_idx += 1
    return result

def _get_accumulator(tensor):
    if False:
        print('Hello World!')
    'Returns TensorList if any containing accumulated values of tensor.\n\n  We try to find a pattern of the form:\n\n     input_tl   tensor\n        \\        /\n    (TensorListPushBack)\n            |\n        output_tl\n\n  which satisfies the following conditions:\n\n  1. input_tl must be in tensor.graph.inputs.\n  2. output_tl or Identity(output_tl) must be in tensor.graph.outputs.\n  3. tensor.graph.input_index(input_tl) == tensor.graph.output_index(output_t).\n\n  output_tl or Identity(output_tl) (whichever is in tensor.graph.outputs) is\n  returned if such a pattern is found else None is returned.\n\n  Args:\n    tensor: The Tensor to be accumulated.\n\n  Returns:\n    A variant tensor in the same graph as `tensor` or None if no accumulator is\n    found.\n  '
    assert isinstance(tensor.graph, func_graph_module.FuncGraph)

    def get_func_graph_output(t):
        if False:
            while True:
                i = 10
        'Returns t or Identity(t) whichever exists in graph outputs else None.'
        for output in tensor.graph.outputs:
            if output is t:
                return t
        identity_op = t.consumers()[0]
        if identity_op.type == 'Identity' and any((identity_op.outputs[0] is t for t in tensor.graph.outputs)):
            return identity_op.outputs[0]
        return None
    for consumer in tensor.consumers():
        if consumer.type != 'TensorListPushBack':
            continue
        accum_input_idx = -1
        for (accum_input_idx, inp) in enumerate(tensor.graph.inputs):
            if inp is consumer.inputs[0]:
                break
        else:
            continue
        output = get_func_graph_output(consumer.outputs[0])
        if output is None:
            continue
        for (accum_output_idx, out) in enumerate(tensor.graph.outputs):
            if out is output:
                if accum_input_idx == accum_output_idx:
                    return output
                break
    return None
OptimizedReductionOpsCacheKey = collections.namedtuple('OptimizedReductionOpsCacheKey', ['op_type', 'inputs', 'dtypes', 'input_types', 'name', 'attrs', 'op_def', 'compute_device'])

class _WhileBodyGradFuncGraph(util.WhileBodyFuncGraph):
    """FuncGraph for the gradient function of the body of a While op.

  Contains the logic for capturing the tensors from the body of the forward
  While op which is as follows:
  1. If the tensor is of resource type (these are not accumulated):
     a. Ensure that the tensor is a loop invariant, i.e., it exists in both loop
        inputs and outputs at the same index.
     b. Lookup the corresponding resource tensor in the forward outer graph and
        try to capture that.
  2. If the tensor is not of resource type:
     a. Create an accumulator for that tensor and output it from the forward
        pass. Note this also requires adding it as an input to the forward pass.
     b. Capture the accumulator from the forward pass in this FuncGraph. This
        will later be resolved to the correct output of the forward While op.
     c. Pop a value from the captured placeholder and use it as the captured
        value for the forward pass tensor.

  This only allows capturing tensors in the forward graph. A ValueError is
  raised if an attempt is made to capture a tensor not in the forward graph.
  To manually capture a tensor that is not in the forward graph, call `capture`
  with `allowlisted=True`.

  Note: The `captures` dict does not contain the forward tensor since it is not
  directly captured. It contains the accumulator corresponding to this forward
  tensor.

  Attributes:
    while_op_needs_rewrite: True if any non-resource intermediates were
      captured, meaning the forward While op needs to be rewritten to output the
      corresponding accumulators.
    extra_inputs: list of EmptyTensorList tensors to be used as initial input to
    the new accumulators in the forward graph. It may also contain external
    captures of the custom gradient function.
    internal_capture_to_output: dict from a tensor_id(captured placeholder) to
      the corresponding tensor that needs to be added to the list of outputs.
      For instance, when capturing an accumulator TensorList this contains the
      TensorList obtained after popping a tensor from the list. Other entries
      in this dict are expected, though not enforced, to be identities.
      This dict is needed because these output tensors need to be added to
      FuncGraph.outputs "after" the tensors returned from the gradient function.
  """

    def __init__(self, name, forward_cond_graph, forward_body_graph, maximum_iterations, forward_while_op, body_graph_inputs, body_graph_outputs):
        if False:
            return 10
        super(_WhileBodyGradFuncGraph, self).__init__(name)
        self.extra_inputs = []
        self.internal_capture_to_output = {}
        self._forward_graph = forward_body_graph
        self._forward_cond_graph = forward_cond_graph
        self._maximum_iterations = maximum_iterations
        self._forward_while_op = forward_while_op
        self._indirect_captures = {}

    @property
    def while_op_needs_rewrite(self):
        if False:
            return 10
        return self.extra_inputs

    def _create_op_internal(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        if False:
            return 10
        optimized_reduction_ops = {'Shape', 'Size', 'Rank', 'TensorListElementShape', 'TensorListLength'}
        if op_type in optimized_reduction_ops and (not util.output_all_intermediates()) and all((input.graph is self._forward_graph for input in inputs)) and all((_get_accumulator(input) is None for input in inputs)) and (not util_v1.GraphOrParentsInXlaContext(self._forward_graph)) and (not util.graph_wrapped_for_higher_order_tape_gradients(self._forward_graph)):
            return self._move_op_to_forward_graph(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)
        return super(_WhileBodyGradFuncGraph, self)._create_op_internal(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)

    def _move_op_to_forward_graph(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        if False:
            print('Hello World!')
        if not hasattr(self._forward_graph, '_optimized_reduction_ops_cache'):
            self._forward_graph._optimized_reduction_ops_cache = {}
        cache_key = self._get_optimized_reduction_ops_cache_key(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)
        cached_op = self._forward_graph._optimized_reduction_ops_cache.get(cache_key)
        if cached_op is not None:
            return cached_op
        with self._forward_graph.as_default():
            name = ops.name_from_scope_name(name)
            result = self._forward_graph._create_op_internal(op_type, inputs, dtypes=dtypes, input_types=input_types, name=name, attrs=attrs, op_def=op_def, compute_device=compute_device)
            self._forward_graph._optimized_reduction_ops_cache[cache_key] = result
            return result

    def _get_optimized_reduction_ops_cache_key(self, op_type, inputs, dtypes=None, input_types=None, name=None, attrs=None, op_def=None, compute_device=True):
        if False:
            while True:
                i = 10
        inputs = tuple(map(lambda t: t.ref(), inputs))
        if dtypes is not None:
            dtypes = tuple(dtypes)
        if input_types is not None:
            input_types = tuple(input_types)
        if attrs is not None:
            hashable_attrs = []
            for (attr_name, attr_value) in sorted(attrs.items()):
                hashable_attrs.append((attr_name, attr_value.SerializeToString()))
            attrs = tuple(hashable_attrs)
        if op_def is not None:
            op_def = op_def.SerializeToString()
        return OptimizedReductionOpsCacheKey(op_type, inputs, dtypes, input_types, name, attrs, op_def, compute_device)

    def _capture_helper(self, tensor, name):
        if False:
            return 10
        'Implements the capturing described in the class docstring.'
        captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
        if captured_tensor is not None:
            return captured_tensor
        if tensor.graph is not self._forward_graph:
            already_captured = id(tensor) in self.function_captures.by_val_internal
            captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)
            if not already_captured:
                self.internal_capture_to_output[ops.tensor_id(captured_tensor)] = captured_tensor
                self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
            return captured_tensor
        while tensor.op.type == 'Identity':
            tensor = tensor.op.inputs[0]
        captured_tensor = self._indirect_captures.get(ops.tensor_id(tensor))
        if captured_tensor is not None:
            return captured_tensor
        if _is_loop_invariant(tensor, self._forward_graph.inputs, self._forward_graph.outputs):
            captured_tensor = super(_WhileBodyGradFuncGraph, self)._capture_helper(tensor, name)
            self.internal_capture_to_output[ops.tensor_id(captured_tensor)] = captured_tensor
            self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
            return captured_tensor
        if constant_op.is_constant(tensor):
            real_value = constant_op.constant(tensor_util.constant_value(tensor), dtype=tensor.dtype)
            self._indirect_captures[ops.tensor_id(tensor)] = real_value
            return real_value
        if tensor.dtype == dtypes.resource:
            return self._resource_capture_helper(tensor)
        accumulator = _get_accumulator(tensor)
        if accumulator is None:
            with self._forward_graph.outer_graph.as_default():
                with util.clear_control_inputs():
                    tensor_list = list_ops.empty_tensor_list(element_dtype=tensor.dtype, element_shape=tensor.shape, max_num_elements=self._maximum_iterations, name=_build_accumulator_name(tensor))
            self.extra_inputs.append(tensor_list)
            with self._forward_graph.as_default():
                accumulator = list_ops.tensor_list_push_back(tensor_list, tensor)
            self._forward_graph.outputs.append(accumulator)
            with self._forward_cond_graph.as_default():
                self._forward_cond_graph.capture(tensor_list)
        captured_accumulator = super(_WhileBodyGradFuncGraph, self)._capture_helper(accumulator, name)
        (new_tensor_list, captured_tensor) = list_ops.tensor_list_pop_back(captured_accumulator, element_dtype=tensor.dtype)
        self._indirect_captures[ops.tensor_id(tensor)] = captured_tensor
        self.internal_capture_to_output[ops.tensor_id(captured_accumulator)] = new_tensor_list
        return captured_tensor

    def _resource_capture_helper(self, tensor):
        if False:
            print('Hello World!')
        'Returns the captured resource tensor.\n\n    Resource-type tensors are not accumulated. If a resource tensor exists in\n    the loop body it must either be a loop input or an output of a nested While\n    op inside the loop body which had captured the external resource.\n\n    Args:\n      tensor: the external resource Tensor to be captured.\n\n    Returns:\n      Tensor in this graph.\n    '
        assert tensor.dtype == dtypes.resource
        forward_graph_input_names = [t.name for t in self._forward_graph.inputs]
        forward_graph_name_to_opdef = {op.name: op.node_def for op in self._forward_graph.get_operations()}
        index = util.resource_input_index(tensor.name, forward_graph_input_names, forward_graph_name_to_opdef, self._forward_graph._functions)
        input_placeholder = self._forward_graph.inputs[index]
        tensor_in_outer_graph = self._forward_graph._while.inputs[index]
        assert input_placeholder.dtype == dtypes.resource
        assert tensor_in_outer_graph.dtype == dtypes.resource
        if index != util.resource_input_index(self._forward_graph.outputs[index].name, forward_graph_input_names, forward_graph_name_to_opdef, self._forward_graph._functions):
            raise AssertionError(f'Resource tensors must be loop invariants {tensor_in_outer_graph}')
        self._indirect_captures[ops.tensor_id(tensor)] = self.capture(tensor_in_outer_graph)
        return self._indirect_captures[ops.tensor_id(tensor)]

def _check_shapes_compat(flat_output_tensors, flat_shape_invariants, flat_input_tensors):
    if False:
        return 10
    for (t, shape, input_t) in zip(flat_output_tensors, flat_shape_invariants, flat_input_tensors):
        if not control_flow_ops._ShapeLessThanOrEqual(t.shape, shape):
            raise ValueError(f'Input tensor `{input_t.name}` enters the loop with shape {shape}, but has shape {t.shape} after one iteration. To allow the shape to vary across iterations, use the `shape_invariants` argument of tf.while_loop to specify a less-specific shape.')

def _check_num_inputs_outputs(cond_graph, body_graph, num_flattened_loop_vars):
    if False:
        i = 10
        return i + 15
    'Checks the number of inputs/outputs of `cond_graph` and `body_graph`.'
    assert len(cond_graph.inputs) == num_flattened_loop_vars, 'cond_graph takes %d inputs; Expected: %d' % (len(cond_graph.inputs), num_flattened_loop_vars)
    assert len(cond_graph.outputs) == 1, 'cond_graph has %d outputs; Expected: 1' % len(cond_graph.outputs)
    assert len(body_graph.inputs) == num_flattened_loop_vars, 'body_graph takes %d inputs; Expected: %d' % (len(body_graph.inputs), num_flattened_loop_vars)
    assert len(body_graph.outputs) == num_flattened_loop_vars, 'body_graph has %d outputs; Expected: %d' % (len(body_graph.outputs), num_flattened_loop_vars)

def _check_inputs_outputs_types_match(body_graph, flattened_loop_vars):
    if False:
        print('Hello World!')
    for (inp, out, loop_var) in zip(body_graph.inputs, body_graph.outputs, flattened_loop_vars):
        if inp.dtype != out.dtype:
            raise TypeError(f'Loop var {loop_var.name} enters the loop with type {inp.dtype} but has type {out.dtype} after 1 iteration. {loop_var.name} type should remain constant.')

def _build_cond_placeholders_name_prefix(cond_graph):
    if False:
        while True:
            i = 10
    return cond_graph.unique_name(cond_graph.name + '___redundant_placeholder')

def _duplicate_body_captures_in_cond(cond_graph, body_graph_captures):
    if False:
        for i in range(10):
            print('nop')
    'Creates placeholders for body captures in cond_graph.\n\n  This is needed to match signatures of cond and body graphs.\n\n  Args:\n    cond_graph: cond branch graph\n    body_graph_captures: Tensors which were captured when building the\n      `body_graph`.\n  '
    types = [t.dtype.as_datatype_enum for t in body_graph_captures]
    with cond_graph._c_graph.get() as c_graph:
        placeholders = c_api.TF_CreatePlaceholders(c_graph, types, compat.as_str(_build_cond_placeholders_name_prefix(cond_graph)))
    placeholder_ops = [ops.Operation._from_c_op(ph.oper, cond_graph) for ph in placeholders]
    tensors = []
    for op in placeholder_ops:
        tensors.append(op.outputs[0])
    tuples = zip(body_graph_captures, tensors)
    keys = [id(t) for t in body_graph_captures]
    for (k, v) in zip(keys, tuples):
        cond_graph._function_captures.add_or_replace(key=k, external=v[0], internal=v[1], is_by_ref=False)
    cond_graph.inputs.extend(tensors)

def _copy_handle_data(src_tensors, tgt_tensors):
    if False:
        while True:
            i = 10
    for (src_t, tgt_t) in zip(src_tensors, tgt_tensors):
        handle_data_util.copy_handle_data(src_t, tgt_t)

def _pack_sequence_as(loop_vars_signature, flat_orig_loop_vars, loop_vars):
    if False:
        return 10
    'Like `nest.pack_sequence_as` but also replaces flows with TensorArrays.'

    def flow_to_tensor_array(flow, ta):
        if False:
            while True:
                i = 10
        return tensor_array_ops.build_ta_with_new_flow(ta, flow) if isinstance(ta, tensor_array_ops.TensorArray) else flow
    flattened_loop_vars = [flow_to_tensor_array(*z) for z in zip(nest.flatten(loop_vars, expand_composites=True), flat_orig_loop_vars)]
    return nest.pack_sequence_as(loop_vars_signature, flattened_loop_vars, expand_composites=True)

def _tensor_array_to_flow(loop_vars):
    if False:
        while True:
            i = 10

    def f(maybe_ta):
        if False:
            for i in range(10):
                print('nop')
        if isinstance(maybe_ta, tensor_array_ops.TensorArray):
            return maybe_ta.flow
        return maybe_ta
    return nest.map_structure(f, loop_vars, expand_composites=True)

def _build_maximum_iterations_loop_var(maximum_iterations):
    if False:
        return 10
    if maximum_iterations is None:
        maximum_iterations = -1
    return ops.convert_to_tensor(maximum_iterations, dtype=dtypes.int32, name='maximum_iterations')

def _build_accumulator_name(tensor):
    if False:
        for i in range(10):
            print('nop')
    return '{}/accumulator'.format(tensor.name).replace(':', '_')

def _is_loop_invariant(tensor, inputs, outputs):
    if False:
        print('Hello World!')
    return any((tensor is t for t in inputs)) and any((tensor is t for t in outputs))

def _set_read_only_resource_inputs_attr(op: ops.Operation, branch_graphs):
    if False:
        print('Hello World!')
    'Sets the list of resource inputs which are read-only.\n\n  This is used by AutomaticControlDependencies.\n\n  Args:\n    op: While Operation.\n    branch_graphs: List of branch FuncGraphs.\n  '
    read_only_indices = set(range(len(op.inputs)))
    for branch_graph in branch_graphs:
        if not read_only_indices:
            break
        branch_read_only_indices = acd.get_read_only_resource_input_indices_graph(branch_graph)
        read_only_indices = read_only_indices.intersection(branch_read_only_indices)
    ops.set_int_list_attr(op, acd.READ_ONLY_RESOURCE_INPUTS_ATTR, sorted(read_only_indices))