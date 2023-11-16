"""Special functions that only make sense for AutoGraph.

These functions are meant to ensure feature parity between Python and AutoGraph,
so that the exact same code works in both modes. In general, AutoGraph will
replace these calls.
"""
from tensorflow.python.autograph.operators import data_structures
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import tensor_util

def _validate_list_constructor(elements, element_dtype, element_shape):
    if False:
        print('Hello World!')
    'Validates the inputs of tensor_list.'
    if element_dtype is not None and element_shape is not None:
        return
    if tensor_util.is_tf_type(elements):
        return
    if isinstance(elements, (list, tuple)):
        if elements:
            return
        else:
            raise ValueError('element_dtype and element_shape are required when elements are empty')
    raise ValueError('unknown type for elements: {}; only Tensor, list and tuple are allowed'.format(type(elements)))

def match_staging_level(value, like_value):
    if False:
        i = 10
        return i + 15
    'Casts a value to be staged at the same level as another.'
    if tensor_util.is_tf_type(like_value):
        return constant_op.constant(value)
    return value

def tensor_list(elements, element_dtype=None, element_shape=None, use_tensor_array=False):
    if False:
        print('Hello World!')
    'Creates an tensor list and populates it with the given elements.\n\n  This function provides a more uniform access to tensor lists and tensor\n  arrays, and allows optional initialization.\n\n  Note: this function is a simplified wrapper. If you need greater control,\n  it is recommended to use the underlying implementation directly.\n\n  Args:\n    elements: Iterable[tf.Tensor, ...], the elements to initially fill the list\n        with\n    element_dtype: Optional[tf.DType], data type for the elements in the list;\n        required if the list is empty\n    element_shape: Optional[tf.TensorShape], shape for the elements in the list;\n        required if the list is empty\n    use_tensor_array: bool, whether to use the more compatible but restrictive\n        tf.TensorArray implementation\n  Returns:\n    Union[tf.Tensor, tf.TensorArray], the new list.\n  Raises:\n    ValueError: for invalid arguments\n  '
    _validate_list_constructor(elements, element_dtype, element_shape)
    if use_tensor_array:
        return data_structures.tf_tensor_array_new(elements, element_dtype, element_shape)
    else:
        return data_structures.tf_tensor_list_new(elements, element_dtype, element_shape)

def stack(list_or_tensor, element_dtype=None, strict=True):
    if False:
        print('Hello World!')
    'Stacks the input, if it admits the notion of stacking.\n\n  For example, a list of tensors can be stacked into a larger tensor. This\n  function is similar to tf.stack, but it accepts non-lists and lists of\n  non-tensors as arguments. In the latter case, the function does nothing.\n\n  Args:\n    list_or_tensor: Any\n    element_dtype: tf.DType, optional dtypedtype for the elements in the list.\n        Required if the input is stackable, and the list is untyped.\n    strict: bool, if True an error is raised if the input is not stackable.\n        Otherwise the function is a no-op.\n\n  Returns:\n    Any, if the input is stackable, the result will be a tf.Tensor. Otherwise,\n    if strict=False, the result will be list_or_tensor.\n\n  Raises:\n    ValueError: if strict=True and the input is not stackable.\n  '
    if strict:

        def raise_error(x):
            if False:
                for i in range(10):
                    print('nop')
            raise ValueError('%s must be stackable when strict=True' % x)
        original_call = raise_error
    else:
        original_call = lambda x: x
    return data_structures.list_stack(list_or_tensor, data_structures.ListStackOpts(element_dtype=element_dtype, original_call=original_call))