"""Implementation for ConcreteFunction."""
import collections
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python import pywrap_tfe
from tensorflow.python.eager import backprop_util
from tensorflow.python.eager import context
from tensorflow.python.eager import forwardprop_util
from tensorflow.python.eager import record
from tensorflow.python.eager.graph_only_ops import graph_placeholder
from tensorflow.python.eager.polymorphic_function import atomic_function
from tensorflow.python.eager.polymorphic_function import attributes as attributes_lib
from tensorflow.python.eager.polymorphic_function import function_type_utils
from tensorflow.python.eager.polymorphic_function import saved_model_exported_concrete
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import func_graph as func_graph_module
from tensorflow.python.framework import indexed_slices
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor as tensor_lib
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import type_spec
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import default_gradient
from tensorflow.python.ops import gradients_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler import trace
from tensorflow.python.trackable import base as trackable
from tensorflow.python.types import core
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import compat
from tensorflow.python.util import nest
from tensorflow.python.util import object_identity

def _is_type_subset(a, b):
    if False:
        while True:
            i = 10
    'Returns true if `b` is a subset of type `a` (or if a is not a TypeSpec.)'
    if isinstance(a, type_spec.TypeSpec):
        return a.most_specific_compatible_type(b) == a
    return True
_FORWARD_PREFIX = '__forward_'
_BACKWARD_PREFIX = '__backward_'
_INFERENCE_PREFIX = '__inference_'

def _forward_name(n):
    if False:
        i = 10
        return i + 15
    'The name of a generated forward defun named n.'
    return '%s%s_%s' % (_FORWARD_PREFIX, n, ops.uid())

def _backward_name(n):
    if False:
        return 10
    'The name of a generated backward defun named n.'
    return '%s%s_%s' % (_BACKWARD_PREFIX, n, ops.uid())

def _inference_name(n):
    if False:
        return 10
    'The name of a forward-but-no-gradient defun named n.'
    return '%s%s_%s' % (_INFERENCE_PREFIX, n, ops.uid())

def _create_forward_backward_with_graph(attrs, forward_graph, backwards_graph: func_graph_module.FuncGraph):
    if False:
        print('Hello World!')
    'Creates forward and backward functions from the function graphs.'
    forward_function_name = _forward_name(forward_graph.name)
    common_attributes = dict(attrs)
    common_attributes.pop(attributes_lib.IMPLEMENTS, None)
    backward_function_attr = attributes_lib.parse_func_attrs({attributes_lib.FORWARD_FUNCTION: forward_function_name})
    backward_function_attr.update(common_attributes)
    function_type = function_type_lib.from_structured_signature(((), {}), backwards_graph.structured_outputs, backwards_graph.function_captures.capture_types)
    backward_function = ConcreteFunction.from_func_graph(backwards_graph, function_type, attrs=backward_function_attr)
    forward_function_attr = attributes_lib.parse_func_attrs({attributes_lib.BACKWARD_FUNCTION: backward_function.name})
    forward_function_attr.update(common_attributes)
    forward_function = atomic_function.from_func_graph(forward_function_name, forward_graph, forward_function_attr)
    return (forward_function, backward_function)

class _DelayedRewriteGradientFunctions(object):
    """Caches forward/backward functions with a delayed forward rewrite."""

    def __init__(self, atomic_fn: atomic_function.AtomicFunction, func_graph_deleter):
        if False:
            return 10
        'Construct an inference function and initialize caches.'
        self._cached_function_pairs = {}
        self._func_graph = atomic_fn.graph
        self._inference_function = atomic_fn
        self._attrs = atomic_fn.attributes
        self._gradient_name = None
        self._num_inference_outputs = len(self._func_graph.outputs)
        self._func_graph_deleter = func_graph_deleter

    def forward_backward(self, num_doutputs=None):
        if False:
            for i in range(10):
                print('nop')
        'A possibly-cached pair of forward and backward functions.'
        if num_doutputs is None:
            num_doutputs = self._num_inference_outputs
        forward_backward = self._cached_function_pairs.get(num_doutputs)
        if forward_backward is not None:
            return forward_backward
        (forward, backward) = self._construct_forward_backward(num_doutputs)
        self._cached_function_pairs[num_doutputs] = (forward, backward)
        return (forward, backward)

    def _construct_forward_backward(self, num_doutputs):
        if False:
            while True:
                i = 10
        'Constructs a pair of forward and backward functions.\n\n    Args:\n      num_doutputs: The constructed backprop function will take output gradients\n        for the first `num_doutputs` outputs of the forward function. Defaults\n        to the number of outputs for the inference function, but when\n        higher-order gradients are computed this will increase to include side\n        outputs.\n\n    Returns:\n      A pair of (forward_function, backward_function):\n        forward_function: A re-generated inference function (an\n          AtomicFunction) to account for new side outputs, if any extra\n          were required when building the backward pass.\n        backward_function: A ConcreteFunction that Takes `num_doutputs`\n          arguments and returns gradients with respect to inputs of the forward\n          function.\n    '
        trainable_outputs = [output for output in self._func_graph.outputs[:num_doutputs] if backprop_util.IsTrainable(output)]
        signature = []
        for t in trainable_outputs:
            signature.append(tensor_lib.TensorSpec(*default_gradient.shape_and_dtype(t)))

        def _backprop_function(*grad_ys):
            if False:
                for i in range(10):
                    print('nop')
            with ops.device(None):
                return gradients_util._GradientsHelper(trainable_outputs, self._func_graph.inputs, grad_ys=grad_ys, src_graph=self._func_graph)
        with self._func_graph.as_default():
            backwards_graph = func_graph_module.FuncGraph(_backward_name(self._func_graph.name))
            func_graph_module.func_graph_from_py_func(name=backwards_graph.name, python_func=_backprop_function, args=[], kwargs={}, signature=signature, func_graph=backwards_graph)
            backwards_graph_captures = backwards_graph.external_captures
            captures_from_forward = [c for c in backwards_graph_captures if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]
            existing_outputs = object_identity.ObjectIdentitySet(self._func_graph.outputs)
            for capture in captures_from_forward:
                if capture not in existing_outputs:
                    existing_outputs.add(capture)
                    self._func_graph.outputs.append(capture)
            (forward_function, backward_function) = _create_forward_backward_with_graph(self._attrs, self._func_graph, backwards_graph)
            return (forward_function, backward_function)

    def _rewrite_forward_and_call_backward(self, op: ops.Operation, *doutputs):
        if False:
            for i in range(10):
                print('nop')
        'Add outputs to the forward call and feed them to the grad function.'
        (forward_function, backwards_function) = self.forward_backward(len(doutputs))
        if not backwards_function.outputs:
            return backwards_function.structured_outputs
        op.graph._add_function_recursive(forward_function)
        op._set_func_attr('f', forward_function.name)
        op._set_type_list_attr('Tout', [o.dtype.as_datatype_enum for o in forward_function.function_type.flat_outputs])
        truncated_outputs = forward_function.function_type.flat_outputs[len(op.outputs):]
        op._add_outputs([o.dtype.as_datatype_enum for o in truncated_outputs], [o.shape for o in truncated_outputs])
        for i in range(len(op.outputs)):
            output_type = forward_function.function_type.flat_outputs[i]
            handle_data = output_type.dtype._handle_data
            if handle_data:
                handle_data_util.set_handle_data(op.outputs[i], handle_data.shape_inference)
        capture_mapping = dict(zip((ops.tensor_id(t) for t in self._func_graph.outputs), op.outputs))
        remapped_captures = [capture_mapping.get(ops.tensor_id(capture), capture) for capture in backwards_function.captured_inputs]
        cleaned_doutputs = []
        for (doutput, placeholder) in zip(doutputs, self._func_graph.outputs):
            if backprop_util.IsTrainable(placeholder):
                if isinstance(doutput, indexed_slices.IndexedSlices):
                    cleaned_doutputs.append(ops.convert_to_tensor(doutput))
                elif doutput is not None:
                    cleaned_doutputs.append(doutput)
                else:
                    cleaned_doutputs.append(default_gradient.zeros_like(placeholder))
        return backwards_function._call_flat(cleaned_doutputs, remapped_captures)

    def get_gradient_function(self):
        if False:
            print('Hello World!')
        "Returns gradient function.\n\n    The gradient rewrites an inference call op to a forward call op, but does\n    not modify a pre-existing forward call op. It then computes the gradient\n    from the output's gradients and the side outputs of the forward op.\n    "
        return self._rewrite_forward_and_call_backward

    def forward(self, inference_args=None, input_tangents=None):
        if False:
            print('Hello World!')
        'A forward function with only user-specified outputs.\n\n    The call operation for the returned inference function can be rewritten into\n    a forward function. This only happens if the backward function (from the\n    `backward` method) ends up being used to compute gradients.\n\n    This approach avoids constructing unnecessary graphs, but it only works if\n    we are calling this function when not executing eagerly.\n\n    Args:\n      inference_args: A flat list of Tensors, arguments to the inference\n        function. Unused, but taken for compatibility with\n        _TapeGradientFunctions.\n      input_tangents: A flat list of Tensors, jvps associated with\n        `inference_args`. Unused; if required, tape functions must be used\n        instead.\n\n    Returns:\n      An atomic_function.AtomicFunction.\n    '
        del inference_args
        if input_tangents:
            raise errors.InternalError('unexpectedly got forwardprop information in a class that does not support forwardprop.')
        return self._inference_function

    def _backward(self, outputs):
        if False:
            print('Hello World!')
        'Fetch a backward function for `outputs` from the forward function.'

        def _backward_function(*args):
            if False:
                return 10
            call_op = outputs[0].op
            return self._rewrite_forward_and_call_backward(call_op, *args)
        return (_backward_function, outputs)

    def record(self, flat_outputs, inference_args, input_tangents):
        if False:
            print('Hello World!')
        'Record the function call operation.\n\n    _DelayedRewriteGradientFunctions supports only first-order backprop tape\n    gradients (and then only when graph building). It does not work with\n    higher-order tape gradients or forward autodiff, but does work with\n    higher-order symbolic gradients (tf.gradients).\n\n    Args:\n      flat_outputs: The result of running `forward`.\n      inference_args: A flat list of Tensors with inference inputs to the\n        operation.\n      input_tangents: A flat list of Tensors with input tangents consumed by the\n        operation.\n    '
        (backward_function, to_record) = self._backward(flat_outputs)
        record.record_operation(self._inference_function.cached_definition.signature.name, to_record, inference_args + input_tangents, backward_function)
_ForwardWrapper = collections.namedtuple('_ForwardWrapper', ('graph', 'outputs', 'output_indices', 'output_tangents'))

class _TapeGradientFunctions(object):
    """Caches forward and backward functions compatible with eager gradients.

  In contrast to the delayed-rewrite approach in
  `_DelayedRewriteGradientFunctions` which only works with delayed execution,
  the forward function generated by this class has a fixed set of outputs which
  may be preserved by a tape in order to compute gradients later.

  This class is abstract; its child classes differ in how many side outputs of
  the forward function their backward function accepts gradients for, which
  determines whether higher-order tape gradients are possible.
  """

    def __init__(self, func_graph: func_graph_module.FuncGraph, attrs, func_graph_deleter, forwardprop_input_indices, delayed_rewrite_functions, need_gradients_for_jvps):
        if False:
            while True:
                i = 10
        self._func_graph = func_graph
        self._forward_graph = None
        self._attrs = attrs
        self._forward = None
        self._backward = None
        self._num_outputs = len(func_graph.outputs)
        self._func_graph_deleter = func_graph_deleter
        self._forwardprop_input_indices = forwardprop_input_indices
        self._forwardprop_output_indices = None
        self._num_forwardprop_outputs = 0
        self._num_inference_outputs = len(func_graph.outputs)
        self._num_trainable_inference_outputs = len([t for t in func_graph.outputs if backprop_util.IsTrainable(t)])
        self._delayed_rewrite_functions = delayed_rewrite_functions
        self._need_gradients_for_jvps = need_gradients_for_jvps

    def _build_functions_for_outputs(self, outputs, inference_args, input_tangents):
        if False:
            i = 10
            return i + 15
        'Forward+backward functions where the backward function sees `outputs`.'
        trainable_outputs = []
        trainable_indices = []
        for (index, output) in enumerate(outputs):
            if backprop_util.IsTrainable(output):
                trainable_outputs.append(output)
                trainable_indices.append(index)
        backwards_graph = func_graph_module.FuncGraph(_backward_name(self._func_graph.name))
        with backwards_graph.as_default():
            gradients_wrt_outputs = []
            for output in trainable_outputs:
                (gradient_shape, gradient_dtype) = default_gradient.shape_and_dtype(output)
                gradient_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
                handle_data_util.copy_handle_data(output, gradient_placeholder)
                gradients_wrt_outputs.append(gradient_placeholder)
            with ops.device(None):
                gradients_wrt_inputs = gradients_util._GradientsHelper(trainable_outputs, self._func_graph.inputs, grad_ys=gradients_wrt_outputs, src_graph=self._func_graph)
            if input_tangents:
                gradients_wrt_inputs = nest.map_structure(lambda x: ops.convert_to_tensor(x) if x is not None else None, gradients_wrt_inputs)
            captures_from_forward = [c for c in backwards_graph.external_captures if not isinstance(c, ops.EagerTensor) and c.graph is self._func_graph]
            existing_outputs = object_identity.ObjectIdentitySet(self._func_graph.outputs)
            for capture in captures_from_forward:
                if capture not in existing_outputs:
                    existing_outputs.add(capture)
                    self._func_graph.outputs.append(capture)
        backwards_graph.inputs = gradients_wrt_outputs + backwards_graph.internal_captures
        backwards_graph.outputs.extend((grad for grad in nest.flatten(gradients_wrt_inputs, expand_composites=True) if grad is not None))
        backwards_graph.structured_outputs = gradients_wrt_inputs
        (forward_function, backward_function) = _create_forward_backward_with_graph(self._attrs, self._func_graph, backwards_graph)
        if not input_tangents:
            return (forward_function, self._func_graph, backward_function, None, 0)
        forward_wrapper = self._wrap_forward_function_with_jvps(forward_function, backward_function, inference_args, input_tangents)
        (wrapped_backwards_graph, forward_wrapper) = self._wrap_backward_function_with_jvp_backprop(backward_function, gradients_wrt_outputs, forward_wrapper)
        forward_wrapper = self._shuffle_forward_outputs(forward_wrapper)
        (wrapped_forward_function, wrapped_backward_function) = _create_forward_backward_with_graph(self._attrs, forward_wrapper.graph, wrapped_backwards_graph)
        if len(inference_args) + len(input_tangents) != len(forward_wrapper.graph.inputs):
            raise errors.InternalError(f'The forward graph had {len(forward_wrapper.graph.inputs)} inputs, but we expected {len(inference_args) + len(input_tangents)} ({len(inference_args)} inference inputs and {len(input_tangents)} input tangents).')
        return (wrapped_forward_function, forward_wrapper.graph, wrapped_backward_function, forward_wrapper.output_indices, len(forward_wrapper.output_tangents))

    def _wrap_forward_function_with_jvps(self, forward_function, backward_function, inference_args, input_tangents):
        if False:
            i = 10
            return i + 15
        'Adds inline JVP computation to a forward function.'
        forward_wrapper_graph = func_graph_module.FuncGraph(_forward_name(self._func_graph.name))
        with forward_wrapper_graph.as_default():
            with forwardprop_util.push_forwardprop_state():
                forward_captures = {ops.tensor_id(internal): external for (external, internal) in self._func_graph.captures}
                for (input_index, real_input) in enumerate(self._func_graph.inputs):
                    input_placeholder = array_ops.placeholder(dtype=real_input.dtype, shape=real_input.shape)
                    capture = forward_captures.get(ops.tensor_id(real_input))
                    if capture is not None:
                        forward_wrapper_graph.add_capture(capture, input_placeholder)
                        if capture.dtype == dtypes.resource:
                            handle_data_util.copy_handle_data(capture, input_placeholder)
                    else:
                        forward_wrapper_graph.inputs.append(input_placeholder)
                for (inp, arg) in zip(forward_wrapper_graph.inputs, inference_args):
                    record.record_operation('captured_value', [inp], [arg], backward_function=lambda x: [x], forward_function=lambda x: [x])
                num_inference_inputs = len(inference_args)
                for tape_indices in self._forwardprop_input_indices:
                    for (input_index, jvp_index) in tape_indices:
                        input_placeholder = forward_wrapper_graph.inputs[input_index]
                        if len(forward_wrapper_graph.inputs) != jvp_index:
                            raise errors.InternalError(f'Expected {jvp_index} forward graph inputs, got {len(forward_wrapper_graph.inputs)}.')
                        (gradient_shape, gradient_dtype) = default_gradient.shape_and_dtype(input_placeholder)
                        jvp_placeholder = graph_placeholder(gradient_dtype, gradient_shape)
                        external_jvp = input_tangents[jvp_index - num_inference_inputs]
                        forward_wrapper_graph.add_capture(external_jvp, jvp_placeholder)
                        tensor_shape.TensorShape(external_jvp.shape).assert_is_compatible_with(jvp_placeholder.shape)
                        record.record_operation('captured_value', [jvp_placeholder], [external_jvp], backward_function=lambda x: [x], forward_function=lambda x: [x])
                forward_inputs = forward_wrapper_graph.inputs[:num_inference_inputs]
                gradient_function = self._delayed_rewrite_functions._rewrite_forward_and_call_backward
                with ops.get_default_graph()._override_gradient_function({'PartitionedCall': gradient_function, 'StatefulPartitionedCall': gradient_function}):
                    forward_outputs = forward_function.call_flat(*forward_inputs)
                    if isinstance(forward_outputs, ops.Operation):
                        forward_outputs = []
                (py_backward, _) = self._wrap_backward_function(self._func_graph, backward_function, forward_outputs)
            record.record_operation_forwardprop_only(forward_function.cached_definition.signature.name, forward_outputs, forward_inputs, py_backward, None)
            (output_indices, output_tangents) = pywrap_tfe.TFE_Py_PackJVPs(forward_outputs)
            output_tangents = [forward_wrapper_graph.capture(t) for t in output_tangents]
        return _ForwardWrapper(graph=forward_wrapper_graph, outputs=forward_outputs, output_indices=output_indices, output_tangents=output_tangents)

    def _wrap_backward_function_with_jvp_backprop(self, backward_function, gradients_wrt_outputs, forward_wrapper):
        if False:
            print('Hello World!')
        'Wraps `backward_function` to include gradients for JVPs.'
        wrapped_backwards_graph = func_graph_module.FuncGraph(_backward_name(self._func_graph.name))
        with wrapped_backwards_graph.as_default():
            (py_backward, recorded_outputs) = self._wrap_backward_function(self._func_graph, backward_function, forward_wrapper.outputs)
            trainable_index = 0
            forward_doutputs = []
            doutput_args = []
            for output in recorded_outputs:
                if backprop_util.IsTrainable(output):
                    doutput = gradients_wrt_outputs[trainable_index]
                    doutput_placeholder = graph_placeholder(doutput.dtype, doutput.shape)
                    doutput_args.append(doutput_placeholder)
                    forward_doutputs.append(doutput_placeholder)
                    trainable_index += 1
                else:
                    doutput_args.append(None)
            dinputs = py_backward(*doutput_args)
            existing_outputs = object_identity.ObjectIdentitySet(forward_wrapper.outputs + forward_wrapper.output_tangents)
            num_processed_output_tangents = 0
            gradients_wrt_output_tangents = []
            tangent_doutputs = []
            output_tangents = forward_wrapper.output_tangents
            output_indices = forward_wrapper.output_indices
            if self._need_gradients_for_jvps:
                while num_processed_output_tangents != len(output_tangents):
                    for output in output_tangents[num_processed_output_tangents:]:
                        (gradient_shape, gradient_dtype) = default_gradient.shape_and_dtype(output)
                        placeholder = graph_placeholder(gradient_dtype, gradient_shape)
                        gradients_wrt_output_tangents.append(placeholder)
                        tangent_doutputs.append(placeholder)
                    num_processed_output_tangents = len(output_tangents)
                    with ops.device(None):
                        gradients_wrt_inputs = gradients_util._GradientsHelper(output_tangents, forward_wrapper.graph.inputs, grad_ys=gradients_wrt_output_tangents, src_graph=forward_wrapper.graph)
                    dinputs = [backprop_util.AggregateIndexedSlicesGradients((existing, new)) for (existing, new) in zip(dinputs, gradients_wrt_inputs) if existing is not None or new is not None]
                    dinputs.extend(gradients_wrt_inputs[len(dinputs):])
                    captures_from_forward = [c for c in wrapped_backwards_graph.external_captures if not isinstance(c, ops.EagerTensor) and c.graph is forward_wrapper.graph]
                    for capture in captures_from_forward:
                        if capture not in existing_outputs:
                            existing_outputs.add(capture)
                            forward_wrapper.outputs.append(capture)
                    (output_indices, output_tangents) = forwardprop_util.pack_tangents(forward_wrapper.outputs)
                    output_tangents = [forward_wrapper.graph.capture(t) for t in output_tangents]
                    for t in output_tangents:
                        existing_outputs.add(t)
        wrapped_backwards_graph.inputs = forward_doutputs[:self._num_trainable_inference_outputs] + tangent_doutputs + forward_doutputs[self._num_trainable_inference_outputs:] + wrapped_backwards_graph.internal_captures
        wrapped_backwards_graph.structured_outputs = dinputs
        wrapped_backwards_graph.outputs = [t for t in dinputs if t is not None]
        return (wrapped_backwards_graph, forward_wrapper._replace(output_indices=output_indices, output_tangents=output_tangents))

    def _shuffle_forward_outputs(self, forward_wrapper):
        if False:
            while True:
                i = 10
        'Reorders function outputs so captures are last.'

        def _index_map(original):
            if False:
                print('Hello World!')
            if original < self._num_inference_outputs:
                return original
            if original >= len(forward_wrapper.outputs):
                return original - len(forward_wrapper.outputs) + self._num_inference_outputs
            return original + len(forward_wrapper.output_tangents)
        output_indices = nest.map_structure(_index_map, forward_wrapper.output_indices)
        forward_wrapper.graph.outputs = forward_wrapper.outputs[:self._num_inference_outputs] + forward_wrapper.output_tangents + forward_wrapper.outputs[self._num_inference_outputs:]
        return forward_wrapper._replace(output_indices=output_indices)

    def forward(self, inference_args, input_tangents):
        if False:
            for i in range(10):
                print('nop')
        'Construct or fetch a forward function with side-outputs.\n\n    When graph building without a tape active, symbolic gradients rely on\n    regenerating the backward function for higher-order gradients (to account\n    for new side outputs of the rewritten forward function call). Thus there is\n    no fixed backward function for this case. However, when a tape is active\n    (eager or graph building), we generate fixed backward and forward functions\n    at forward function call time.\n\n    This difference between the tape and non-tape cases is to avoid building\n    unneeded backward functions while graph building (where we may or may not\n    eventually need gradients).\n\n    Args:\n      inference_args: A flat list of Tensors, arguments to the inference\n        function.\n      input_tangents: A flat list of Tensors, jvps associated with\n        `inference_args`.\n\n    Returns:\n      A forward atomic_function.AtomicFunction.\n    '
        if self._forward is None:
            (self._forward, self._forward_graph, self._backward, self._forwardprop_output_indices, self._num_forwardprop_outputs) = self._forward_and_backward_functions(inference_args, input_tangents)
        return self._forward

    def _wrap_backward_function(self, forward_graph: func_graph_module.FuncGraph, backward, outputs):
        if False:
            while True:
                i = 10
        'Create a backward function given `outputs` from the forward function.'
        capture_mapping = dict(zip((ops.tensor_id(t) for t in forward_graph.outputs), outputs))
        captured_inputs = backward.captured_inputs
        remapped_captures = [capture_mapping.get(ops.tensor_id(capture), capture) for capture in captured_inputs]
        if any((t.graph is forward_graph for t in remapped_captures if not isinstance(t, ops.EagerTensor))):
            incorrect_mapping = [t for t in remapped_captures if not isinstance(t, ops.EagerTensor) and t.graph is not forward_graph]
            raise errors.InternalError(f'Failed to map all backward graph captures to the forward graph. Incorrectly mapped: {incorrect_mapping}.')
        variant_zeros_like = {}
        backward_function_inputs = len(backward.inputs) - len(captured_inputs)
        recorded_outputs = []
        trainable_recorded_outputs = 0
        skip_positions = []
        if self._num_forwardprop_outputs and (not self._need_gradients_for_jvps):
            relevant_outputs = outputs[:self._num_inference_outputs] + outputs[self._num_inference_outputs + self._num_forwardprop_outputs:]
        else:
            relevant_outputs = outputs
        for (output_index, output) in enumerate(relevant_outputs):
            if trainable_recorded_outputs < backward_function_inputs:
                recorded_outputs.append(output)
            if backprop_util.IsTrainable(output):
                trainable_recorded_outputs += 1
            else:
                skip_positions.append(output_index)
            if output.dtype == dtypes.variant:
                variant_zeros_like[output_index] = default_gradient.zeros_like(output)

        def _backward_function_wrapper(*args):
            if False:
                return 10
            'Process output gradients and call the backward function.'
            if not backward.outputs:
                return backward.structured_outputs
            processed_args = []
            input_index = 0
            for (output_index, arg) in enumerate(args):
                if isinstance(arg, indexed_slices.IndexedSlices):
                    arg = ops.convert_to_tensor(arg)
                if output_index in skip_positions:
                    continue
                if arg is None:
                    input_placeholder = backward.inputs[input_index]
                    if input_placeholder.dtype == dtypes.variant:
                        arg = variant_zeros_like[output_index]
                    else:
                        arg = array_ops.zeros(*default_gradient.shape_and_dtype(input_placeholder))
                processed_args.append(arg)
                input_index += 1
                if input_index >= backward_function_inputs:
                    break
            return backward._call_flat(processed_args, remapped_captures)
        return (_backward_function_wrapper, recorded_outputs)

    def record(self, flat_outputs, inference_args, input_tangents):
        if False:
            print('Hello World!')
        'Record the function call operation.\n\n    For backprop, indicates the backward function to use and which new Tensors\n    must be watched. For forwardprop from eager, the function call itself will\n    have produced tangents which need to be recorded.\n\n    Args:\n      flat_outputs: The result of running `forward`.\n      inference_args: A flat list of Tensors with inference inputs to the\n        operation.\n      input_tangents: A flat list of Tensors with input tangents consumed by the\n        operation.\n    '
        (backward_function, to_record) = self._wrap_backward_function(self._forward_graph, self._backward, flat_outputs)
        if self._forwardprop_output_indices:
            record.record_operation_backprop_only(self._forward.cached_definition.signature.name, to_record, inference_args, backward_function)
            record.record_operation_forwardprop_only(self._forward.cached_definition.signature.name, flat_outputs, inference_args + input_tangents, backward_function, self._forwardprop_output_indices)
        else:
            record.record_operation(self._forward.cached_definition.signature.name, to_record, inference_args + input_tangents, backward_function)

class _FirstOrderTapeGradientFunctions(_TapeGradientFunctions):
    """Caches tape-friendly functions for first-order gradients."""

    def __init__(self, func_graph: func_graph_module.FuncGraph, attrs, func_graph_deleter, forwardprop_input_indices, delayed_rewrite_functions, need_gradients_for_jvps):
        if False:
            while True:
                i = 10
        super().__init__(func_graph, attrs, func_graph_deleter, forwardprop_input_indices, delayed_rewrite_functions, need_gradients_for_jvps)
        self._func_graph_deleter = func_graph_deleter
        self._forwardprop_input_indices = forwardprop_input_indices

    def _forward_and_backward_functions(self, inference_args, input_tangents):
        if False:
            i = 10
            return i + 15
        'Shortcut for when only first-order gradients are required.\n\n    The returned backward function does not accept gradients with respect to\n    side output of forward_function. This is fine as long as the user can\'t\n    possibly request second order tape gradients, as when they\'ve used a single\n    non-persistent GradientTape. Since we don\'t need the backward function to\n    take gradients with respect to side outputs, we can skip some potentially\n    slow graph building.\n\n    Args:\n      inference_args: A flat list of Tensors, arguments to the inference\n        function.\n      input_tangents: A flat list of Tensors, jvps associated with\n        `inference_args`.\n\n    Returns:\n      A tuple of (forward_function, backward_function):\n        forward_function: Takes the same inputs as the inference function, but\n          returns side outputs used by backward_function in addition to the\n          inference function\'s outputs.\n        backward_function: Takes side outputs from forward_function and\n          gradients with respect to the "real" outputs of forward_function and\n          returns gradients with respect to the inputs.\n    '
        outputs = self._func_graph.outputs[:self._num_inference_outputs]
        return self._build_functions_for_outputs(outputs, inference_args, input_tangents)

class _HigherOrderTapeGradientFunctions(_TapeGradientFunctions):
    """Caches tape-friendly functions for higher-order gradients."""

    def _forward_and_backward_functions(self, inference_args, input_tangents):
        if False:
            i = 10
            return i + 15
        "Forward and backward functions suitable for higher-order gradients.\n\n    Unlike in `_FirstOrderTapeGradientFunctions`, the backward function built by\n    this method accepts gradients for all of the outputs of the returned forward\n    function, including side outputs.\n\n    Args:\n      inference_args: A flat list of Tensors, arguments to the inference\n        function.\n      input_tangents: A flat list of Tensors, jvps associated with\n        `inference_args`.\n\n    Returns:\n      A tuple of (forward_function, backward_function):\n        forward_function: Takes the same inputs as the inference function, but\n          returns side outputs used by backward_function in addition to the\n          inference function's outputs.\n        backward_function: Takes side outputs from forward_function and\n          gradients with respect to all of its outputs, real and side. Returns\n          gradients with respect to the inputs.\n    "
        outputs = []
        iteration_count = 0
        while len(outputs) < len(self._func_graph.outputs) and any((backprop_util.IsTrainable(output) for output in self._func_graph.outputs[len(outputs):])):
            iteration_count += 1
            if iteration_count >= 20 and iteration_count % 5 == 0:
                new_op_with_trainable_output = None
                num_new_trainable_outputs = 0
                for output in self._func_graph.outputs[len(outputs):]:
                    if backprop_util.IsTrainable(output):
                        num_new_trainable_outputs += 1
                        new_op_with_trainable_output = output.op
                logging.warning("Determining side outputs for the function '{}' is taking longer than expected ({} iterations, typically this converges in 5 or so). This could indicate that a gradient registration is adding new ops to the forward pass every time gradients are generated. {} new trainable output(s) were added this iteration, one from the following op:\n {}\nThis may indicate a TensorFlow bug, or an issue in a tf.custom_gradient.".format(self._func_graph.name, iteration_count, num_new_trainable_outputs, new_op_with_trainable_output))
            outputs = list(self._func_graph.outputs)
            self._build_functions_for_outputs(outputs, inference_args, input_tangents)
        (forward_function, forward_graph, backward_function, output_indices, num_output_tangents) = self._build_functions_for_outputs(outputs, inference_args, input_tangents)
        if len(self._func_graph.outputs) > len(outputs) and any((backprop_util.IsTrainable(output) for output in self._func_graph.outputs[len(outputs):])):
            raise errors.InternalError(f'Unexpectedly added new outputs to the forward function when building the backward function: {self._func_graph.outputs[len(outputs):]}.')
        return (forward_function, forward_graph, backward_function, output_indices, num_output_tangents)

class _ForwardBackwardCall(object):
    """Holds the state of a function call between execution and recording."""
    __slots__ = ['_functions', '_inference_args', '_input_tangents', '_tape_watching']

    def __init__(self, functions, inference_args, input_tangents, tape_watching):
        if False:
            for i in range(10):
                print('nop')
        'Collects information about the function call.\n\n    Args:\n      functions: An object which produces forward and backward functions, either\n        a _DelayedRewriteGradientFunctions or a _TapeGradientFunctions object.\n      inference_args: A flat list of Tensors, arguments to the inference\n        function.\n      input_tangents: A flat list of Tensors, jvps associated with\n        `inference_args`.\n      tape_watching: Boolean, with True indicating that recording is necessary.\n    '
        self._functions = functions
        self._inference_args = inference_args
        self._input_tangents = input_tangents
        self._tape_watching = tape_watching

    def forward(self):
        if False:
            print('Hello World!')
        'Builds or retrieves a forward function for this call.'
        forward_function = self._functions.forward(self._inference_args, self._input_tangents)
        return (forward_function, self._inference_args + self._input_tangents)

    def record(self, flat_outputs):
        if False:
            while True:
                i = 10
        'Given outputs from the execution of `forward`, records the operation.'
        if self._tape_watching and (not isinstance(flat_outputs, ops.Operation)) and (flat_outputs is not None):
            self._functions.record(flat_outputs, self._inference_args, self._input_tangents)

class ConcreteFunction(core.ConcreteFunction, trackable.Trackable):
    """A `tf.types.experimental.ConcreteFunction` created from `tf.function`."""

    def __init__(self, atomic_fn: atomic_function.AtomicFunction, shared_func_graph=True):
        if False:
            while True:
                i = 10
        'Initialize a `ConcreteFunction`.\n\n    Args:\n     atomic_fn: Inference atomic function to form basis of forward pass.\n     shared_func_graph: If False, the ConcreteFunction takes ownership of\n       `func_graph` and will break reference cycles when it is deleted. This\n       makes the FuncGraph inoperable.\n\n    Raises:\n      ValueError: If number of input_placeholders is not equal to the number\n        of function inputs.\n    '
        self._arg_keywords = None
        self._num_positional_args = None
        self._func_graph = atomic_fn.graph
        self._captured_inputs = self._func_graph.external_captures + self._func_graph.deferred_external_captures
        self._function_type = atomic_fn.function_type
        self._output_shapes = tuple((output.shape for output in self._func_graph.outputs))
        self._attrs = attributes_lib.parse_func_attrs(atomic_fn.attributes or {})
        if shared_func_graph:
            self._garbage_collector = None
        else:
            self._garbage_collector = ConcreteFunctionGarbageCollector(atomic_fn.graph)
        self._delayed_rewrite_functions = _DelayedRewriteGradientFunctions(atomic_fn, self._garbage_collector)
        self._first_order_tape_functions = {}
        self._higher_order_tape_functions = {}
        self._inference_function = self._delayed_rewrite_functions.forward()

    @classmethod
    def from_func_graph(cls, graph, function_type, attrs, shared_func_graph=True):
        if False:
            print('Hello World!')
        atomic_fn = atomic_function.from_func_graph(_inference_name(graph.name), graph, attrs, function_type)
        return ConcreteFunction(atomic_fn, shared_func_graph=shared_func_graph)

    @property
    def function_type(self):
        if False:
            i = 10
            return i + 15
        'Return the FunctionType associated with this ConcreteFunction.'
        return self._function_type

    @property
    def inference_fn(self):
        if False:
            return 10
        'Return the inference function associated with this ConcreteFunction.'
        return self._inference_function

    @property
    def _function_spec(self):
        if False:
            while True:
                i = 10
        if self.function_type is None:
            return None
        return function_type_utils.FunctionSpec(self.function_type, {p.default for p in self.function_type.parameters.values() if p.optional}, False, name=self.name)

    @property
    def variables(self):
        if False:
            return 10
        'Sequence of variables for this function.'
        return tuple(self._func_graph.variables)

    def set_variables(self, variables):
        if False:
            i = 10
            return i + 15
        self._func_graph.variables = variables

    @property
    def trainable_variables(self):
        if False:
            return 10
        'Sequence of trainable variables for this function.'
        return tuple(self._func_graph.trainable_variables)

    def __call__(self, *args, **kwargs):
        if False:
            return 10
        "Executes the wrapped function.\n\n    ConcreteFunctions have two signatures:\n\n    * The signature of the original function wrapped by this ConcreteFunction.\n    * A flat signature, where each argument accepts a single Tensor.\n\n    The original function signature is generally preferred, but the flat input\n    signature is supported for backward compatibility.\n\n    ### Original Function Signature\n\n    When calling a ConcreteFunction with the signature of the original function,\n    each argument must match the type or value that was used when the\n    ConcreteFunction's graph was traced.  In particular:\n\n    * Tensor arguments (including CompositeTensors, such as RaggedTensor) must\n      have matching `TypeSpec`s.\n    * Non-Tensor arguments (such as booleans or ints) must have equal values.\n    * Nested arguments (such as lists, tuples, or dictionaries) must have the\n      same nesting structure; and each nested value must have a matching type\n      or value.\n\n    The default value for any arguments that were traced with non-Tensor values\n    is the value that was used in the trace.  Arguments that were traced with\n    tensor arguments do not have a default value (even if the original function\n    had a default value for that argument).\n\n    ### Flat Signature\n\n    When calling a ConcreteFunction with the flat signature, the arguments\n    correspond to the flattened component tensors of the arguments that were\n    used to construct the ConcreteFunction.  Parameter names are assigned based\n    on `TensorSpec.name` (when specified) or the original argument names (with\n    suffixes automatically added for nested arguments or composite tensors with\n    multiple components).\n\n    Args:\n      *args: Positional arguments to the concrete function.\n      **kwargs: Keyword arguments to the concrete function.\n\n    Returns:\n      The result of applying the TF function on the given Tensors.\n\n    Raises:\n      AssertionError: If this `ConcreteFunction` was not created through\n        `get_concrete_function`.\n      TypeError: If the arguments do not match the function's signature.\n    "
        return self._call_impl(args, kwargs)

    def _call_impl(self, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        'See `__call__` for details.'
        with trace.Trace(self._func_graph.name, tf_function_call='concrete'):
            if self.function_type is not None:
                try:
                    return self._call_with_structured_signature(args, kwargs)
                except TypeError as structured_err:
                    try:
                        return self._call_with_flat_signature(args, kwargs)
                    except (TypeError, ValueError) as flat_err:
                        raise TypeError(str(structured_err) + '\nFallback to flat signature also failed due to: ' + str(flat_err))
            return self._call_with_flat_signature(args, kwargs)

    def _call_with_flat_signature(self, args, kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Executes the wrapped function with the flat signature.\n\n    Args:\n      args: Positional arguments to the concrete function.\n      kwargs: Keyword arguments to the concrete function.\n\n    Returns:\n      The result of applying the function on the Tensors/Variables contained in\n      `args` and `kwargs`.\n    Raises:\n      TypeError: if `args` and `kwargs` do not match the flat signature of this\n        `ConcreteFunction`.\n    '
        if len(args) > self._num_positional_args:
            raise TypeError(f'{self._flat_signature_summary()} takes {self._num_positional_args} positional arguments, got {len(args)}.')
        args = list(args)
        kwargs = dict(kwargs)
        kwargs = {function_type_lib.sanitize_arg_name(k): v for (k, v) in kwargs.items()}
        for keyword in self._arg_keywords[len(args):]:
            try:
                args.append(kwargs.pop(function_type_lib.sanitize_arg_name(compat.as_str(keyword))))
            except KeyError:
                specified_keywords = list(self._arg_keywords[:len(args)]) + list(kwargs.keys())
                missing_required_args = sorted(set(self._arg_keywords) - set(specified_keywords))
                raise TypeError(f"{self._flat_signature_summary()} missing required arguments: {', '.join(missing_required_args)}.")
        if kwargs:
            positional_arg_keywords = set(self._arg_keywords[:len(args)])
            for unused_key in kwargs:
                if unused_key in positional_arg_keywords:
                    raise TypeError(f"{self._flat_signature_summary()} got two values for '{unused_key}'.")
            raise TypeError(f"{self._flat_signature_summary()} got unexpected keyword arguments: {', '.join(sorted(kwargs))}.")
        for (i, arg) in enumerate(args):
            if not isinstance(arg, (tensor_lib.Tensor, resource_variable_ops.BaseResourceVariable)):
                raise TypeError(f'{self._flat_signature_summary()}: expected argument #{i}(zero-based) to be a Tensor; got {type(arg).__name__} ({arg}).')
        return self._call_flat(args, self.captured_inputs)

    def _call_with_structured_signature(self, args, kwargs):
        if False:
            while True:
                i = 10
        'Executes the wrapped function with the structured signature.\n\n    Args:\n      args: Positional arguments to the concrete function.\n      kwargs: Keyword arguments to the concrete function.\n\n    Returns:\n      The result of applying the function on the Tensors/Variables contained in\n      `args` and `kwargs`.\n    Raises:\n      TypeError: if `args` and `kwargs` do not match the structured signature\n        of this `ConcreteFunction`.\n    '
        bound_args = function_type_utils.canonicalize_function_inputs(args, kwargs, self.function_type)
        filtered_flat_args = self.function_type.unpack_inputs(bound_args)
        return self._call_flat(filtered_flat_args, captured_inputs=self.captured_inputs)

    def _call_flat(self, tensor_inputs, captured_inputs):
        if False:
            print('Hello World!')
        'Executes the wrapped function.\n\n    Args:\n      tensor_inputs: a list of only Tensors generated from args, kwargs.\n      captured_inputs: the captured inputs that are also part of the input args\n        to the actual execution. By default, it should be self._captured_inputs.\n    Returns:\n      The result of applying the TF function to `args`.\n\n    Raises:\n      ValueError: If `args` contains anything other than Tensors or Variables.\n    '
        ctx = context.context()
        executing_eagerly = ctx.executing_eagerly()
        default_graph = ops.get_default_graph()
        if default_graph.building_function and (not self._func_graph.saveable):
            default_graph.mark_as_unsaveable(self._func_graph.saving_errors)
        if record.could_possibly_record() or hasattr(default_graph, 'watch_variable'):
            for v in self._func_graph.variables:
                resource_variable_ops.variable_accessed(v)
        if not executing_eagerly:
            for (i, tensor_input) in enumerate(tensor_inputs):
                if tensor_input.dtype == dtypes.resource or tensor_input.dtype == dtypes.variant:
                    continue
                graph_input_shape = tensor_shape.TensorShape(self._func_graph.inputs[i].shape)
                if not graph_input_shape.is_compatible_with(tensor_input.shape):
                    raise ValueError(f'Tensor {tensor_input} is not compatible with the shape this function was traced with. Expected shape {self._func_graph.inputs[i].shape}, but got shape {tensor_input.shape}.\n\nIf you called get_concrete_function, you may need to pass a tf.TensorSpec(..., shape=...) with a less specific shape, having None on axes which can vary.')
        args = tensor_inputs + captured_inputs
        possible_gradient_type = gradients_util.PossibleTapeGradientTypes(args)
        if possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_NONE and executing_eagerly:
            return self._inference_function.call_preflattened(args)
        forward_backward = self._select_forward_and_backward_functions(args, possible_gradient_type, executing_eagerly)
        (forward_function, args_with_tangents) = forward_backward.forward()
        if executing_eagerly:
            flat_outputs = forward_function.call_flat(*args_with_tangents)
        else:
            with default_graph._override_gradient_function({'PartitionedCall': self._get_gradient_function(), 'StatefulPartitionedCall': self._get_gradient_function()}):
                flat_outputs = forward_function.call_flat(*args_with_tangents)
        forward_backward.record(flat_outputs)
        return self.function_type.pack_output(flat_outputs)

    @property
    def name(self):
        if False:
            while True:
                i = 10
        '`ConcreteFunction` name.'
        return self._delayed_rewrite_functions.forward().name

    @property
    def graph(self):
        if False:
            while True:
                i = 10
        'Returns the graph from which this function was constructed.'
        return self._func_graph

    @property
    def inputs(self):
        if False:
            return 10
        'Returns tensors in `self.graph` corresponding to arguments.'
        return self._func_graph.inputs

    @property
    def structured_input_signature(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns structured signature for this concrete function.\n\n    Returns:\n      A tuple `(args, kwargs)`, where:\n\n        * `args` is a tuple that specifies the expected type or value each for\n          positional argument.\n        * `kwargs` is a dictionary that specifies the expected type or value\n          for each keyword-only argument.\n\n      The type or value for each argument is specified using one of the\n      following:\n\n        * A `tf.TypeSpec`, indicating that a Tensor or other TensorFlow-native\n          value is expected.\n        * A Python value, such as an integer, indicating that an equal value\n          is expected.\n        * A nested structure of `tf.TypeSpec`s and Python values, indicating\n          that a corresponding nested structure is expected.\n    '
        return self._func_graph.structured_input_signature

    @property
    def outputs(self):
        if False:
            while True:
                i = 10
        'Returns tensors in `self.graph` corresponding to returned tensors.'
        return self._func_graph.outputs

    @property
    def structured_outputs(self):
        if False:
            return 10
        'Returns outputs in `self.graph` as returned by the original function.'
        return self._func_graph.structured_outputs

    def set_external_captures(self, captures):
        if False:
            print('Hello World!')
        'Updates the function capture values.\n\n    The new values must have tensor types and shapes consistent with the\n    original captures of the concrete function, but it is allowed to change a\n    value captured with a deferred one and vice-versa.\n\n    Args:\n      captures: A list of tensors or closures. Tensors are value captures, and\n        closures are call-time (deferred captures).\n    '
        self._captured_inputs = captures

    def replace_capture_with_deferred_capture(self, tensor, closure, spec, placeholder=None, default_value=None):
        if False:
            i = 10
            return i + 15
        "Replaces existing capture `tensor` with a deferred capture `closure`.\n\n    This API replaces the capture `tensor` from the concrete function's captured\n    inputs list, and places the deferred capture `closure` in\n    its spot so the order of captured inputs is preserved. This is important\n    because the old `tensor` and the new `closure` will have the same internal\n    placeholder, which can be passed through the `placeholder` argument, or\n    skipped, in which case we find the placeholder from internal inputs by\n    indexing `tensor` in the external captured inputs list. Thus, it is\n    important that the new deferred capture has output spec (specified by the\n    `spec` argument) compatible with the internal placeholder (`placeholder`)\n    and the original capture (`tensor`).\n\n    For example,\n\n    ```python\n    bool_captured_tensor = tf.constant(True)\n    float_captured_tensor = tf.constant([3.], dtype=tf.float32)\n    value = tf.constant([2.], dtype=tf.float32)\n\n    @tf.function\n    def fn():\n      deferred_tensor = ops.get_default_graph().capture_call_time_value(\n          lambda: value,\n          tf.TensorSpec(shape=(1,), dtype=tf.float32))\n      if bool_captured_tensor:\n        return deferred_tensor\n      else:\n        return deferred_tensor + float_captured_tensor\n\n    concrete_fn = fn.get_concrete_function()\n    print(concrete_fn())  # tf.Tensor([2.], shape=(1,), dtype=float32)\n\n    new_bool_captured_tensor = constant_op.constant(False)\n    def bool_closure():\n      return new_bool_captured_tensor\n\n    concrete_fn.replace_capture_with_deferred_capture(\n        bool_captured_tensor,\n        bool_closure,\n        spec=tensor_lib.TensorSpec(shape=(), dtype=dtypes.bool))\n\n    print(concrete_fn())  # tf.Tensor([5.], shape=(1,), dtype=float32)\n    ```\n\n    Args:\n      tensor: Tensor already captured. This `tensor` should be listed in\n        concrete_function.captured_inputs except when it's empty such as when\n        the concrete function is restored from SavedModel.\n      closure: function which takes no arguments, to be evaluated at function\n        call time, returning a nest of tensors compatible with `spec`.\n      spec: nest of TypeSpec for the value to capture.\n      placeholder: optional. The internal placeholder corresponding to the\n        captured `tensor` and the new `closure`.\n      default_value: optional value to use in environments that cannot safely\n        evaluate closure.\n    "
        capture_index = None
        for (i, capture) in enumerate(self._captured_inputs):
            if id(tensor) == id(capture):
                capture_index = i
                break
        if placeholder is None:
            if capture_index is None:
                raise ValueError(f"Did not find `tensor` argument {tensor} in the ConcreteFunction's captured inputs list, and did not receive a placeholder argument. Thus we're unable to infer the internal placeholder. ")
            placeholder = self.inputs[-len(self._captured_inputs) + capture_index]
        if not (spec.is_compatible_with(tensor) or spec.is_compatible_with(placeholder)):
            raise ValueError(f"Attempting to substitute closure with spec {spec} that's incompatible with the original capture {tensor} or the internal placeholder {placeholder}.")
        self._func_graph.replace_capture_with_deferred_capture(tensor=tensor, closure=closure, spec=spec, placeholder=placeholder, default_value=default_value)
        if capture_index is not None:
            self._captured_inputs[capture_index] = closure

    @property
    def captured_inputs(self):
        if False:
            return 10
        'Returns external Tensors captured by this function.\n\n    self.__call__(*args) passes `args + self.captured_inputs` to the function.\n    '
        return nest.flatten([x() if callable(x) else x for x in self._captured_inputs], expand_composites=True)

    @property
    def function_def(self):
        if False:
            print('Hello World!')
        'Returns a `FunctionDef` object representing this function.'
        return self._delayed_rewrite_functions.forward().cached_definition

    @property
    def output_shapes(self):
        if False:
            return 10
        "The function's output shapes."
        return nest.map_structure(lambda x: getattr(x, 'shape', tensor_shape.TensorShape(None)), composite_tensor.replace_composites_with_components(self._func_graph.structured_outputs), expand_composites=False)

    @property
    def output_dtypes(self):
        if False:
            i = 10
            return i + 15
        return nest.map_structure(lambda x: x.dtype if x is not None else None, composite_tensor.replace_composites_with_components(self._func_graph.structured_outputs), expand_composites=False)

    def add_to_graph(self, g=None, overwrite=False):
        if False:
            for i in range(10):
                print('nop')
        'Registers the function, adds it to the graph g or default graph.\n\n    Args:\n      g: If specified, registers the function with this graph. Defaults to the\n        current context (either the default graph or the eager context).\n      overwrite: A bool. If True, its forward function will overwrite\n        any existing function of the same signature name in the graph `g`.\n    '
        if not context.executing_eagerly() and (not g):
            g = ops.get_default_graph()
        if g is not None:
            g._add_function_recursive(self._delayed_rewrite_functions.forward())

    def add_gradient_functions_to_graph(self, g=None):
        if False:
            while True:
                i = 10
        'Add forward/backward functions to graph `g` or the current context.'
        if not context.executing_eagerly() and (not g):
            g = ops.get_default_graph()
        g._add_function_recursive(self._delayed_rewrite_functions.forward())
        (forward_function, backward_function) = self._delayed_rewrite_functions.forward_backward()
        g._add_function_recursive(forward_function)
        backward_function.add_to_graph(g)

    def _get_gradient_function(self):
        if False:
            return 10
        'Returns gradient function. It will be lazily created at first call.'
        return self._delayed_rewrite_functions._rewrite_forward_and_call_backward

    def _select_forward_and_backward_functions(self, args, possible_gradient_type, executing_eagerly):
        if False:
            while True:
                i = 10
        'Selects forward and backward functions based on the calling context.\n\n    The forward function computes the "real" function outputs, `self._outputs`,\n    and any extra values needed by the corresponding backward function.\n\n    Args:\n      args: A flat list of Tensors with all of the inputs to the forward\n        function (including user-specified and captured inputs).\n      possible_gradient_type: One of gradients_util.POSSIBLE_GRADIENT_TYPES_*.\n      executing_eagerly: Boolean, the value of context.executing_eagerly().\n\n    Returns:\n      An object with a `forward` method returning a tuple of (forward_function :\n      AtomicFunction, augmented_arguments : List), and a corresponding\n      `record` method which takes outputs from the forward function and records\n      the operation. forward_function should be called with augmented_arguments.\n    '
        if executing_eagerly:
            input_tangents = forwardprop_util.pack_tangents(args)
        else:
            input_tangents = forwardprop_util.TangentInfo()
        need_gradients_for_jvps = record.should_record_backprop(input_tangents.tangents)
        cache_key = (need_gradients_for_jvps, input_tangents.indices)
        if possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_FIRST_ORDER:
            if input_tangents.indices or executing_eagerly:
                functions = self._first_order_tape_functions.get(cache_key, None)
                if functions is None:
                    functions = _FirstOrderTapeGradientFunctions(self._func_graph, self._attrs, self._garbage_collector, forwardprop_input_indices=input_tangents.indices, delayed_rewrite_functions=self._delayed_rewrite_functions, need_gradients_for_jvps=need_gradients_for_jvps)
                    self._first_order_tape_functions[cache_key] = functions
                return _ForwardBackwardCall(functions, args, input_tangents.tangents, tape_watching=True)
            else:
                return _ForwardBackwardCall(self._delayed_rewrite_functions, args, input_tangents.tangents, tape_watching=True)
        elif possible_gradient_type == gradients_util.POSSIBLE_GRADIENT_TYPES_HIGHER_ORDER:
            functions = self._higher_order_tape_functions.get(cache_key, None)
            if functions is None:
                functions = _HigherOrderTapeGradientFunctions(self._func_graph, self._attrs, self._garbage_collector, forwardprop_input_indices=input_tangents.indices, delayed_rewrite_functions=self._delayed_rewrite_functions, need_gradients_for_jvps=need_gradients_for_jvps)
                self._higher_order_tape_functions[cache_key] = functions
            return _ForwardBackwardCall(functions, args, input_tangents.tangents, tape_watching=True)
        return _ForwardBackwardCall(self._delayed_rewrite_functions, args, input_tangents.tangents, tape_watching=False)

    @property
    def _as_name_attr_list(self):
        if False:
            print('Hello World!')
        'Returns a `NameAttrList` representing this function.'
        ret = attr_value_pb2.NameAttrList(name=self.name)
        for (name, value) in self._attrs.items():
            ret.attr[name].CopyFrom(value)
        return ret

    def _flat_signature_summary(self):
        if False:
            return 10
        "Returns a string summarizing this function's flat signature."
        assert self._arg_keywords is not None
        assert self._num_positional_args is not None
        arg_names = self._arg_keywords
        if self._num_positional_args > len(arg_names):
            arg_names.extend(('<arg{}>'.format(i + 1) for i in range(len(arg_names), self._num_positional_args)))
        return f"{self._func_graph.name}({', '.join(arg_names)})"

    def pretty_printed_signature(self, verbose=True):
        if False:
            i = 10
            return i + 15
        'Returns a string summarizing the signature of this concrete function.'
        assert self.function_type is not None
        if verbose:
            return repr(self.function_type)
        else:
            return str(self.function_type)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if self.function_type is not None:
            return '<ConcreteFunction {} at 0x{:X}>'.format(self.pretty_printed_signature(verbose=False), id(self))
        elif not (self._num_positional_args is None or self._arg_keywords is None):
            return '<ConcreteFunction {} at 0x{:X}>'.format(self._flat_signature_summary(), id(self))
        else:
            return object.__repr__(self)

    def __str__(self):
        if False:
            i = 10
            return i + 15
        if self.function_type is not None:
            return 'ConcreteFunction {}'.format(self.pretty_printed_signature(verbose=True))
        else:
            return self.__repr__()

    def _trackable_children(self, save_type='checkpoint', **kwargs):
        if False:
            i = 10
            return i + 15
        'Implements `Trackable`.'
        if save_type == 'checkpoint':
            return {}
        captured_trackables = {}
        for (n, (capture, _)) in enumerate(self.graph.captures):
            if capture.dtype not in (dtypes.variant, dtypes.resource) and (not resource_variable_ops.is_resource_variable(capture)):
                captured_trackables[f'capture_{n}'] = capture
        return captured_trackables

    def _deserialization_dependencies(self, children):
        if False:
            while True:
                i = 10
        return children

    def _export_to_saved_model_graph(self, object_map, tensor_map, **unused_kwargs):
        if False:
            print('Hello World!')
        if not self.graph.saveable:
            raise ValueError(f'Unable to save function {self.name} for the following reason(s):\n' + '\n'.join(self.graph.saving_errors))
        self.add_to_graph()
        object_map[self] = saved_model_exported_concrete.ExportedConcreteFunction(self, tensor_map)
        return []
_pywrap_utils.RegisterType('Tensor', tensor_lib.Tensor)
_pywrap_utils.RegisterType('EagerTensor', ops.EagerTensor)
_pywrap_utils.RegisterType('IndexedSlices', indexed_slices.IndexedSlices)

class ConcreteFunctionGarbageCollector:
    """Cleans up reference cycles when a `ConcreteFunction` goes out of scope."""
    __slots__ = ['_func_graph']

    def __init__(self, func_graph):
        if False:
            i = 10
            return i + 15
        self._func_graph = func_graph

    def release(self):
        if False:
            i = 10
            return i + 15
        'Call off the FuncGraph deletion.'
        self._func_graph = None

    def __del__(self):
        if False:
            return 10
        if func_graph_module is None or self._func_graph is None:
            return
        try:
            func_graph_module.dismantle_func_graph(self._func_graph)
        except:
            pass

class _Marker(object):
    """Markers used to pretty-print nested args in function signatures."""
    __slots__ = ['_s']

    def __init__(self, s):
        if False:
            while True:
                i = 10
        self._s = s

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return str(self._s)

def _contains_type_spec(value):
    if False:
        i = 10
        return i + 15
    return any((isinstance(x, type_spec.TypeSpec) for x in nest.flatten(value)))