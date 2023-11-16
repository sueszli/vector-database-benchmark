"""Utility functions for manipulating Keras models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf

def extract_submodel(model, inputs, outputs, name=None):
    if False:
        i = 10
        return i + 15
    "Extracts a section of a Keras model into a new model.\n\n  This method walks an existing model from the specified outputs back to the\n  specified inputs in order to construct a new model containing only a portion\n  of the old model, while sharing the layers and weights with the original\n  model.\n\n  WARNING: This method does not work for submodels containing layers that have\n  been used multiple times in the original model, or in other models beyond\n  the original model. (E.g. does not work for submodels that contain layers that\n  use shared weights). This also means that multiple overlapping submodels\n  cannot be extracted from the same model.\n\n  It also relies on recursion and will hit python's recursion limit for large\n  submodels.\n\n  Args:\n    model: The existing Keras model this method extracts a submodel from.\n    inputs: The layer inputs in the existing model that start the submodel\n    outputs: The layer outputs in the existing model that should be output by\n      the submodel\n    name: The name for the extracted model\n\n  Returns:\n    The extracted submodel specified by the given inputs and outputs\n  "
    output_to_layer = {}
    output_to_layer_input = {}
    for layer in model.layers:
        layer_output = layer.output
        layer_inputs = layer.input
        output_to_layer[layer_output] = layer
        output_to_layer_input[layer_output] = layer_inputs
    model_inputs_dict = {}
    memoized_results = {}

    def _recurse_in_model(tensor):
        if False:
            i = 10
            return i + 15
        'Walk the existing model recursively to copy a submodel.'
        if tensor in memoized_results:
            return memoized_results[tensor]
        if tensor == inputs or (isinstance(inputs, list) and tensor in inputs):
            if tensor not in model_inputs_dict:
                model_inputs_dict[tensor] = tf.keras.layers.Input(tensor=tensor)
            out = model_inputs_dict[tensor]
        else:
            cur_inputs = output_to_layer_input[tensor]
            cur_layer = output_to_layer[tensor]
            if isinstance(cur_inputs, list):
                out = cur_layer([_recurse_in_model(inp) for inp in cur_inputs])
            else:
                out = cur_layer(_recurse_in_model(cur_inputs))
        memoized_results[tensor] = out
        return out
    if isinstance(outputs, list):
        model_outputs = [_recurse_in_model(tensor) for tensor in outputs]
    else:
        model_outputs = _recurse_in_model(outputs)
    if isinstance(inputs, list):
        model_inputs = [model_inputs_dict[tensor] for tensor in inputs]
    else:
        model_inputs = model_inputs_dict[inputs]
    return tf.keras.Model(inputs=model_inputs, outputs=model_outputs, name=name)