"""A shim layer for working with functions exported/restored from saved models.

This functionality should ultimately be moved into a first-class core API.
"""
import warnings
from tensorflow.core.function.polymorphism import function_type as function_type_lib
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import handle_data_util
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import variables as variables_lib
from tensorflow.python.trackable import asset
from tensorflow.python.trackable import resource

def get_tensor_from_node(node):
    if False:
        print('Hello World!')
    'Resolves a saved model graph node into a tensor to be captured.\n\n  Args:\n    node: a tensor, variable, or resource to be resolved into a capturable\n      tensor\n\n  Returns:\n    A list of tensors.\n  Raises:\n    ValueError: if the node cannot be converted into a tensor.\n  '
    with ops.init_scope():
        if getattr(node, 'is_distributed_variable', False):
            return node
        elif getattr(node, 'is_distributed_table', False):
            return node
        elif getattr(node, 'is_sharded_variable', False):
            return node
        elif resource_variable_ops.is_resource_variable(node):
            return node.handle
        elif isinstance(node, asset.Asset):
            return node.asset_path
        elif tensor_util.is_tf_type(node):
            return node
        elif isinstance(node, resource.CapturableResource):
            return node.resource_handle
        raise ValueError(f'Cannot convert node {node} to tensor.')

def restore_captures(concrete_function, inputs):
    if False:
        return 10
    'Restore captures for the concrete function.\n\n  Used at deserialization time.  For functions that are being deserialized,\n  saved model restores objects that tensors were captured from, but functions\n  only know about their tensors -- object information is destroyed by tracing.\n  This additional logic extracts the tensors which the function originally\n  captured.\n\n  Args:\n    concrete_function: the concrete function for which to restore captures\n    inputs: a list tensors or other Python objects (such as variables) which\n      contain tensors that were originally captured by the function\n  '
    bound_inputs = [get_tensor_from_node(obj) for obj in inputs]
    bound_variables = [obj for obj in inputs if isinstance(obj, (variables_lib.Variable, resource_variable_ops.BaseResourceVariable))]
    captured_inputs_list = []
    concrete_function.set_variables(bound_variables)
    if bound_inputs:
        for (bound_input, internal_capture) in zip(bound_inputs, concrete_function.inputs[-len(bound_inputs):]):
            if hasattr(bound_input, '__tf_experimental_restore_capture__'):
                captured_inputs_list.append(bound_input.__tf_experimental_restore_capture__(concrete_function, internal_capture))
            else:
                captured_inputs_list.append(bound_input)
                concrete_function.graph.replace_capture(bound_input, internal_capture)
                if internal_capture.dtype == dtypes.resource:
                    if resource_variable_ops.is_resource_variable(bound_input):
                        try:
                            handle = bound_input.handle
                        except ValueError:
                            pass
                        else:
                            handle_data_util.copy_handle_data(handle, internal_capture)
                    else:
                        handle_data_util.copy_handle_data(bound_input, internal_capture)
                concrete_function.graph.capture(bound_input)
    if any([inp is None for inp in captured_inputs_list]):
        warnings.warn("Trying to load ShardedVariables using tf.saved_model.load. This won't work if using a tf.distribute.Strategy, and may use excess memory if not using a Strategy. Ignore this warning if using tf.keras.models.load_model.")
    concrete_function.set_external_captures(captured_inputs_list)
    if concrete_function.function_type:
        concrete_function._function_type = function_type_lib.FunctionType(concrete_function.function_type.parameters.values(), concrete_function.graph.function_captures.capture_types, return_annotation=concrete_function.function_type.output)