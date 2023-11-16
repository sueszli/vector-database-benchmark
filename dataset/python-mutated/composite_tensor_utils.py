"""Utility to manipulate CompositeTensors in tf.function."""
from tensorflow.python.framework import composite_tensor
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest

def flatten_with_variables(inputs):
    if False:
        print('Hello World!')
    "Flattens `inputs` but don't expand `ResourceVariable`s."
    flat_inputs = []
    for value in nest.flatten(inputs):
        if isinstance(value, composite_tensor.CompositeTensor) and (not _pywrap_utils.IsResourceVariable(value)):
            components = value._type_spec._to_components(value)
            flat_inputs.extend(flatten_with_variables(components))
        else:
            flat_inputs.append(value)
    return flat_inputs