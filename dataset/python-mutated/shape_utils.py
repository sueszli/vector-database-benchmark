"""Utils used to manipulate tensor shapes."""
import tensorflow.compat.v2 as tf

def assert_shape_equal(shape_a, shape_b):
    if False:
        for i in range(10):
            print('nop')
    'Asserts that shape_a and shape_b are equal.\n\n  If the shapes are static, raises a ValueError when the shapes\n  mismatch.\n\n  If the shapes are dynamic, raises a tf InvalidArgumentError when the shapes\n  mismatch.\n\n  Args:\n    shape_a: a list containing shape of the first tensor.\n    shape_b: a list containing shape of the second tensor.\n\n  Returns:\n    Either a tf.no_op() when shapes are all static and a tf.assert_equal() op\n    when the shapes are dynamic.\n\n  Raises:\n    ValueError: When shapes are both static and unequal.\n  '
    if all((isinstance(dim, int) for dim in shape_a)) and all((isinstance(dim, int) for dim in shape_b)):
        if shape_a != shape_b:
            raise ValueError('Unequal shapes {}, {}'.format(shape_a, shape_b))
        else:
            return tf.no_op()
    else:
        return tf.assert_equal(shape_a, shape_b)

def combined_static_and_dynamic_shape(tensor):
    if False:
        print('Hello World!')
    'Returns a list containing static and dynamic values for the dimensions.\n\n  Returns a list of static and dynamic values for shape dimensions. This is\n  useful to preserve static shapes when available in reshape operation.\n\n  Args:\n    tensor: A tensor of any type.\n\n  Returns:\n    A list of size tensor.shape.ndims containing integers or a scalar tensor.\n  '
    static_tensor_shape = tensor.shape.as_list()
    dynamic_tensor_shape = tf.shape(input=tensor)
    combined_shape = []
    for (index, dim) in enumerate(static_tensor_shape):
        if dim is not None:
            combined_shape.append(dim)
        else:
            combined_shape.append(dynamic_tensor_shape[index])
    return combined_shape

def pad_or_clip_nd(tensor, output_shape):
    if False:
        for i in range(10):
            print('nop')
    'Pad or Clip given tensor to the output shape.\n\n  Args:\n    tensor: Input tensor to pad or clip.\n    output_shape: A list of integers / scalar tensors (or None for dynamic dim)\n      representing the size to pad or clip each dimension of the input tensor.\n\n  Returns:\n    Input tensor padded and clipped to the output shape.\n  '
    tensor_shape = tf.shape(input=tensor)
    clip_size = [tf.where(tensor_shape[i] - shape > 0, shape, -1) if shape is not None else -1 for (i, shape) in enumerate(output_shape)]
    clipped_tensor = tf.slice(tensor, begin=tf.zeros(len(clip_size), dtype=tf.int32), size=clip_size)
    clipped_tensor_shape = tf.shape(input=clipped_tensor)
    trailing_paddings = [shape - clipped_tensor_shape[i] if shape is not None else 0 for (i, shape) in enumerate(output_shape)]
    paddings = tf.stack([tf.zeros(len(trailing_paddings), dtype=tf.int32), trailing_paddings], axis=1)
    padded_tensor = tf.pad(tensor=clipped_tensor, paddings=paddings)
    output_static_shape = [dim if not isinstance(dim, tf.Tensor) else None for dim in output_shape]
    padded_tensor.set_shape(output_static_shape)
    return padded_tensor