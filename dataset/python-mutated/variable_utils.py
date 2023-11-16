"""Utility to manipulate resource variables."""
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import ops
from tensorflow.python.util import _pywrap_utils
from tensorflow.python.util import nest

def convert_variables_to_tensors(values):
    if False:
        print('Hello World!')
    'Converts `ResourceVariable`s in `values` to `Tensor`s.\n\n  If an object is a `CompositeTensor` and overrides its\n  `_convert_variables_to_tensors` method, its `ResourceVariable` components\n  will also be converted to `Tensor`s. Objects other than `ResourceVariable`s\n  in `values` will be returned unchanged.\n\n  Args:\n    values: A nested structure of `ResourceVariable`s, or any other objects.\n\n  Returns:\n    A new structure with `ResourceVariable`s in `values` converted to `Tensor`s.\n  '

    def _convert_resource_variable_to_tensor(x):
        if False:
            print('Hello World!')
        if _pywrap_utils.IsResourceVariable(x):
            return ops.convert_to_tensor(x)
        elif isinstance(x, composite_tensor.CompositeTensor):
            return composite_tensor.convert_variables_to_tensors(x)
        else:
            return x
    return nest.map_structure(_convert_resource_variable_to_tensor, values)

def replace_variables_with_atoms(values):
    if False:
        i = 10
        return i + 15
    "Replaces `ResourceVariable`s in `values` with tf.nest atoms.\n\n  This function is mostly for backward compatibility. Historically,\n  `ResourceVariable`s are treated as tf.nest atoms. This is no\n  longer the case after `ResourceVariable` becoming `CompositeTensor`.\n  Unfortunately, tf.nest doesn't allow customization of what objects\n  are treated as atoms. Calling this function to manually convert\n  `ResourceVariable`s to atoms to avoid breaking tf.assert_same_structure\n  with inputs of a `ResourceVariable` and an atom, like a `Tensor`.\n\n  The specific implementation uses 0 as the tf.nest atom, but other tf.nest\n  atoms could also serve the purpose. Note, the `TypeSpec` of None is not a\n  tf.nest atom.\n\n  Objects other than `ResourceVariable`s in `values` will be returned unchanged.\n\n  Note: this function does not look into `CompositeTensor`s. Replacing\n  `ResourceVariable`s in a `CompositeTensor` with atoms will change the\n  `TypeSpec` of the `CompositeTensor`, which violates the semantics of\n  `CompositeTensor` and tf.nest. So `ResourceVariable`s in `CompositeTensor`s\n  will be returned as they are.\n\n  Args:\n    values: A nested structure of `ResourceVariable`s, or any other objects.\n\n  Returns:\n    A new structure with `ResourceVariable`s in `values` converted to atoms.\n  "

    def _replace_resource_variable_with_atom(x):
        if False:
            print('Hello World!')
        if _pywrap_utils.IsResourceVariable(x):
            return 0
        else:
            return x
    return nest.map_structure(_replace_resource_variable_with_atom, values)