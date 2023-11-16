"""Common TF utilities."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import six
import tensorflow as tf
from official.modeling import activations

def pack_inputs(inputs):
    if False:
        for i in range(10):
            print('nop')
    'Pack a list of `inputs` tensors to a tuple.\n\n  Args:\n    inputs: a list of tensors.\n\n  Returns:\n    a tuple of tensors. if any input is None, replace it with a special constant\n    tensor.\n  '
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if x is None:
            outputs.append(tf.constant(0, shape=[], dtype=tf.int32))
        else:
            outputs.append(x)
    return tuple(outputs)

def unpack_inputs(inputs):
    if False:
        for i in range(10):
            print('nop')
    'unpack a tuple of `inputs` tensors to a tuple.\n\n  Args:\n    inputs: a list of tensors.\n\n  Returns:\n    a tuple of tensors. if any input is a special constant tensor, replace it\n    with None.\n  '
    inputs = tf.nest.flatten(inputs)
    outputs = []
    for x in inputs:
        if is_special_none_tensor(x):
            outputs.append(None)
        else:
            outputs.append(x)
    x = tuple(outputs)
    if len(x) == 1:
        return x[0]
    return tuple(outputs)

def is_special_none_tensor(tensor):
    if False:
        return 10
    'Checks if a tensor is a special None Tensor.'
    return tensor.shape.ndims == 0 and tensor.dtype == tf.int32

def get_activation(identifier):
    if False:
        return 10
    'Maps a identifier to a Python function, e.g., "relu" => `tf.nn.relu`.\n\n  It checks string first and if it is one of customized activation not in TF,\n  the corresponding activation will be returned. For non-customized activation\n  names and callable identifiers, always fallback to tf.keras.activations.get.\n\n  Args:\n    identifier: String name of the activation function or callable.\n\n  Returns:\n    A Python function corresponding to the activation function.\n  '
    if isinstance(identifier, six.string_types):
        name_to_fn = {'gelu': activations.gelu, 'simple_swish': activations.simple_swish, 'hard_swish': activations.hard_swish, 'identity': activations.identity}
        identifier = str(identifier).lower()
        if identifier in name_to_fn:
            return tf.keras.activations.get(name_to_fn[identifier])
    return tf.keras.activations.get(identifier)

def get_shape_list(tensor, expected_rank=None, name=None):
    if False:
        print('Hello World!')
    'Returns a list of the shape of tensor, preferring static dimensions.\n\n  Args:\n    tensor: A tf.Tensor object to find the shape of.\n    expected_rank: (optional) int. The expected rank of `tensor`. If this is\n      specified and the `tensor` has a different rank, and exception will be\n      thrown.\n    name: Optional name of the tensor for the error message.\n\n  Returns:\n    A list of dimensions of the shape of tensor. All static dimensions will\n    be returned as python integers, and dynamic dimensions will be returned\n    as tf.Tensor scalars.\n  '
    if expected_rank is not None:
        assert_rank(tensor, expected_rank, name)
    shape = tensor.shape.as_list()
    non_static_indexes = []
    for (index, dim) in enumerate(shape):
        if dim is None:
            non_static_indexes.append(index)
    if not non_static_indexes:
        return shape
    dyn_shape = tf.shape(tensor)
    for index in non_static_indexes:
        shape[index] = dyn_shape[index]
    return shape

def assert_rank(tensor, expected_rank, name=None):
    if False:
        for i in range(10):
            print('nop')
    "Raises an exception if the tensor rank is not of the expected rank.\n\n  Args:\n    tensor: A tf.Tensor to check the rank of.\n    expected_rank: Python integer or list of integers, expected rank.\n    name: Optional name of the tensor for the error message.\n\n  Raises:\n    ValueError: If the expected shape doesn't match the actual shape.\n  "
    expected_rank_dict = {}
    if isinstance(expected_rank, six.integer_types):
        expected_rank_dict[expected_rank] = True
    else:
        for x in expected_rank:
            expected_rank_dict[x] = True
    actual_rank = tensor.shape.ndims
    if actual_rank not in expected_rank_dict:
        raise ValueError('For the tensor `%s`, the actual tensor rank `%d` (shape = %s) is not equal to the expected tensor rank `%s`' % (name, actual_rank, str(tensor.shape), str(expected_rank)))