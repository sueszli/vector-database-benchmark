"""Methods for rewriting while_v2 grad functions with IndexedSlices output."""
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import tensor
from tensorflow.python.framework import tensor_conversion
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.util import nest

def rewrite_grad_indexed_slices(grads, body_grad_graph, loop_vars, forward_inputs):
    if False:
        print('Hello World!')
    'Handles special case of IndexedSlices returned from while gradient.\n\n  Some gradient functions return IndexedSlices instead of a Tensor (e.g. the\n  gradient of Gather ops). When this happens in the gradient of a while body,\n  the resulting gradient body function will have mismatched inputs and outputs,\n  since the input is a single Tensor, but the IndexedSlices gets unnested into\n  three output Tensors.\n\n  This function fixes this by rewriting the gradient body to have three inputs\n  to match the three outputs, i.e., it effectively converts the input Tensor\n  into an input IndexedSlices. It also returns new `loop_vars` to reflect the\n  new inputs.\n\n  Args:\n    grads: the input gradient Tensors to the while gradient computation.\n    body_grad_graph: _WhileBodyGradFuncGraph.\n    loop_vars: list of Tensors. The inputs to body_grad_graph.\n    forward_inputs: list of Tensors. The (flat) inputs to the forward-pass While\n      op.\n\n  Returns:\n    The new loop_vars to pass to body_grad_graph.\n  '
    inputs_with_grads = [t for (g, t) in zip(grads, forward_inputs) if g is not None]
    structured_outputs = body_grad_graph.structured_outputs[3:]
    for (forward_input, output) in zip(inputs_with_grads, structured_outputs):
        if not isinstance(output, indexed_slices.IndexedSlices):
            continue
        if forward_input.dtype == dtypes.resource:
            loop_vars = _rewrite_input_as_indexed_slices(body_grad_graph, output, forward_input, loop_vars)
        else:
            _rewrite_output_as_tensor(body_grad_graph, output)
    return loop_vars

def _get_tensor_index_in_iterable(iterable, t):
    if False:
        for i in range(10):
            print('nop')
    'Returns index of first occurence of `t`, raises ValueError if not found.'
    for (i, elem) in enumerate(iterable):
        if t is elem:
            return i
    raise ValueError(f'Element `{t!r}` is not found in iterable `{iterable!r}`.')

def _rewrite_output_as_tensor(body_grad_graph, grad_output_slices):
    if False:
        print('Hello World!')
    'Rewrites grad_output_slices to be a Tensor output.\n\n  Args:\n    body_grad_graph: _WhileBodyGradFuncGraph.\n    grad_output_slices: IndexedSlices output of body_grad_graph.\n  '
    with body_grad_graph.as_default():
        new_output = tensor_conversion.convert_to_tensor_v2(grad_output_slices)
    idx = _get_tensor_index_in_iterable(body_grad_graph.structured_outputs, grad_output_slices)
    body_grad_graph.structured_outputs[idx] = new_output
    body_grad_graph.outputs = func_graph.flatten(body_grad_graph.structured_outputs)

def _rewrite_input_as_indexed_slices(body_grad_graph, grad_output_slices, forward_input, loop_vars):
    if False:
        print('Hello World!')
    "Rewrites grad_output_slices's corresponding input to be an IndexedSlices.\n\n  This rewrite requires that forward_input was captured in the forward loop,\n  i.e. is not a user-specified loop variable. This is important because the\n  rewrite assumes that forward_input is passed through to its corresponding\n  output unchanged. This assumption is used in _rewrite_input_as_indexed_slices,\n  which depends on the exact gradient structure produced by the input's fanout.\n\n  This can yield a more efficient computation than using\n  _rewrite_output_as_tensor, since it preserves the IndexedSlices structure\n  instead of converting the IndexedSlices to a dense Tensor.\n\n  Args:\n    body_grad_graph: _WhileBodyGradFuncGraph.\n    grad_output_slices: IndexedSlices output of body_grad_graph.\n    forward_input: the corresponding Tensor input to the forward loop.\n    loop_vars: list of Tensors. The inputs to body_grad_graph.\n\n  Returns:\n    The new loop_vars to pass to body_grad_graph.\n  "
    init_slices = _create_grad_indexed_slices_init(grad_output_slices, forward_input)
    with body_grad_graph.as_default():
        input_slices = indexed_slices.IndexedSlices(values=body_grad_graph.capture(init_slices.values, allowlisted=True), indices=body_grad_graph.capture(init_slices.indices, allowlisted=True), dense_shape=body_grad_graph.capture(init_slices.dense_shape, allowlisted=True))
        for t in _flatten(init_slices):
            captured_t = body_grad_graph.captures.pop(t)
            body_grad_graph.inputs.remove(captured_t)
        new_output_slices = _rewrite_grad_indexed_slices_output(grad_output_slices, input_slices)
    return _update_indexed_slices_param(body_grad_graph, loop_vars, init_slices, input_slices, new_output_slices, grad_output_slices)

def _create_grad_indexed_slices_init(grad_output_slices, forward_input):
    if False:
        return 10
    'Creates an IndexedSlices to pass as input to the while grad function.\n\n  Args:\n    grad_output_slices: IndexedSlices. The corresponding while grad function\n      output.\n    forward_input: Tensor. The corresponding input to the forward while op.\n\n  Returns:\n    Zeros IndexedSlices, created in current Graph.\n  '
    assert isinstance(grad_output_slices, indexed_slices.IndexedSlices)
    assert isinstance(forward_input, tensor.Tensor)
    values_out = grad_output_slices.values
    indices_out = grad_output_slices.indices
    if values_out.shape.is_fully_defined():
        values_shape = tensor_shape.TensorShape([0] + values_out.shape.as_list()[1:])
        values = array_ops.zeros(values_shape, dtype=values_out.dtype, name='values_init')
    else:
        if forward_input.dtype == dtypes.resource:
            forward_shape = gen_resource_variable_ops.variable_shape(forward_input)
        else:
            forward_shape = array_ops.shape(forward_input)
        values_shape = array_ops.concat([[0], forward_shape[1:]], 0)
        values = array_ops.zeros(values_shape, dtype=values_out.dtype, name='values_init')
    indices = constant_op.constant([], indices_out.dtype, name='indices_init')
    if forward_input.dtype == dtypes.resource:
        shape = gen_resource_variable_ops.variable_shape(forward_input, name='shape_init')
    else:
        shape = array_ops.shape(forward_input, name='shape_init')
    return indexed_slices.IndexedSlices(values=values, indices=indices, dense_shape=shape)

def _rewrite_grad_indexed_slices_output(old_output_slices, new_input_slices):
    if False:
        for i in range(10):
            print('nop')
    'Creates a new version of old_output_slices with new_input_slices as input.\n\n  This method assumes that old_output_slices.{values,indices} are produced by\n  concatenating the incoming gradient Tensor input with the IndexedSlices\n  produced by the gradient computation of the while body. See\n  backprop.aggregate_indexed_slices_gradients for where these concats are\n  constructed. We build new concats that use new_input_slices instead of the\n  original Tensor input.\n\n  Args:\n    old_output_slices: original IndexedSlices output of while gradient.\n    new_input_slices: new IndexedSlices to use as input to while gradient.\n\n  Returns:\n    A new IndexedSlices to replace old_output_slices.\n  '

    def rewrite(old_output, new_input):
        if False:
            while True:
                i = 10
        assert old_output.type == 'Identity'
        concat_op = old_output.inputs[0].op
        assert concat_op.type == 'ConcatV2'
        old_concat_args = concat_op.inputs[:-1]
        return array_ops.concat([new_input] + old_concat_args[1:], 0)
    values = rewrite(old_output_slices.values.op, new_input_slices.values)
    indices = rewrite(old_output_slices.indices.op, new_input_slices.indices)
    return indexed_slices.IndexedSlices(values=values, indices=indices, dense_shape=new_input_slices.dense_shape)

def _update_indexed_slices_param(graph, loop_vars, init_slices, input_slices, output_slices, old_output_slices):
    if False:
        print('Hello World!')
    "Updates graph with new IndexedSlices input/output.\n\n  Updates graph's metadata to output the gradient computation defined by\n  init_slices, input_slices, and output_slices, instead of outputting\n  old_output_slices. Also returns a new version of loop_vars with init_slices\n  replacing the old input.\n\n  Args:\n    graph: _WhileBodyGradFuncGraph.\n    loop_vars: the inputs to graph.\n    init_slices: the new IndexedSlices to use as input to graph.\n    input_slices: the new IndexedSlices in graph that should be fed by\n      init_slices.\n    output_slices: the new IndexedSlices in graph that should be the\n      corresponding output to input_slices.\n    old_output_slices: the IndexedSlices in graph that are currently being\n      output.\n\n  Returns:\n    New loop_vars to pass to graph.\n  "
    structured_idx = _get_tensor_index_in_iterable(graph.structured_outputs, old_output_slices)
    flat_idx = _get_tensor_index_in_iterable(graph.outputs, func_graph.flatten(old_output_slices)[0])
    graph.structured_outputs[structured_idx] = output_slices
    graph.outputs = func_graph.flatten(graph.structured_outputs)
    graph.inputs = graph.inputs[:flat_idx] + _flatten(input_slices) + graph.inputs[flat_idx + 1:]
    return loop_vars[:flat_idx] + _flatten(init_slices) + loop_vars[flat_idx + 1:]

def _flatten(arg):
    if False:
        print('Hello World!')
    return nest.flatten(arg, expand_composites=True)