import numpy as np
from keras import backend
from keras.api_export import keras_export

@keras_export('keras.utils.normalize')
def normalize(x, axis=-1, order=2):
    if False:
        print('Hello World!')
    "Normalizes an array.\n\n    If the input is a NumPy array, a NumPy array will be returned.\n    If it's a backend tensor, a backend tensor will be returned.\n\n    Args:\n        x: Array to normalize.\n        axis: axis along which to normalize.\n        order: Normalization order (e.g. `order=2` for L2 norm).\n\n    Returns:\n        A normalized copy of the array.\n    "
    from keras import ops
    if not isinstance(order, int) or not order >= 1:
        raise ValueError(f'Argument `order` must be an int >= 1. Received: order={order}')
    if isinstance(x, np.ndarray):
        norm = np.atleast_1d(np.linalg.norm(x, order, axis))
        norm[norm == 0] = 1
        axis = axis or -1
        return x / np.expand_dims(norm, axis)
    if len(x.shape) == 0:
        x = ops.expand_dims(x, axis=0)
    epsilon = backend.epsilon()
    if order == 2:
        power_sum = ops.sum(ops.square(x), axis=axis, keepdims=True)
        norm = ops.reciprocal(ops.sqrt(ops.maximum(power_sum, epsilon)))
    else:
        power_sum = ops.sum(ops.power(x, order), axis=axis, keepdims=True)
        norm = ops.reciprocal(ops.power(ops.maximum(power_sum, epsilon), 1.0 / order))
    return ops.multiply(x, norm)

@keras_export('keras.utils.to_categorical')
def to_categorical(x, num_classes=None):
    if False:
        i = 10
        return i + 15
    'Converts a class vector (integers) to binary class matrix.\n\n    E.g. for use with `categorical_crossentropy`.\n\n    Args:\n        x: Array-like with class values to be converted into a matrix\n            (integers from 0 to `num_classes - 1`).\n        num_classes: Total number of classes. If `None`, this would be inferred\n            as `max(x) + 1`. Defaults to `None`.\n\n    Returns:\n        A binary matrix representation of the input as a NumPy array. The class\n        axis is placed last.\n\n    Example:\n\n    >>> a = keras.utils.to_categorical([0, 1, 2, 3], num_classes=4)\n    >>> print(a)\n    [[1. 0. 0. 0.]\n     [0. 1. 0. 0.]\n     [0. 0. 1. 0.]\n     [0. 0. 0. 1.]]\n\n    >>> b = np.array([.9, .04, .03, .03,\n    ...               .3, .45, .15, .13,\n    ...               .04, .01, .94, .05,\n    ...               .12, .21, .5, .17],\n    ...               shape=[4, 4])\n    >>> loss = keras.backend.categorical_crossentropy(a, b)\n    >>> print(np.around(loss, 5))\n    [0.10536 0.82807 0.1011  1.77196]\n\n    >>> loss = keras.backend.categorical_crossentropy(a, a)\n    >>> print(np.around(loss, 5))\n    [0. 0. 0. 0.]\n    '
    if backend.is_tensor(x):
        return backend.nn.one_hot(x, num_classes)
    x = np.array(x, dtype='int64')
    input_shape = x.shape
    if input_shape and input_shape[-1] == 1 and (len(input_shape) > 1):
        input_shape = tuple(input_shape[:-1])
    x = x.reshape(-1)
    if not num_classes:
        num_classes = np.max(x) + 1
    batch_size = x.shape[0]
    categorical = np.zeros((batch_size, num_classes))
    categorical[np.arange(batch_size), x] = 1
    output_shape = input_shape + (num_classes,)
    categorical = np.reshape(categorical, output_shape)
    return categorical

def encode_categorical_inputs(inputs, output_mode, depth, dtype='float32', count_weights=None, backend_module=None):
    if False:
        i = 10
        return i + 15
    'Encodes categoical inputs according to output_mode.'
    backend_module = backend_module or backend
    if output_mode == 'int':
        return backend_module.cast(inputs, dtype=dtype)
    original_shape = inputs.shape
    if len(backend_module.shape(inputs)) == 0:
        inputs = backend_module.numpy.expand_dims(inputs, -1)
    if len(backend_module.shape(inputs)) > 2:
        raise ValueError(f"When output_mode is not `'int'`, maximum supported output rank is 2. Received output_mode {output_mode} and input shape {original_shape}, which would result in output rank {inputs.shape.rank}.")
    binary_output = output_mode in ('multi_hot', 'one_hot')
    if binary_output:
        if output_mode == 'one_hot':
            bincounts = backend_module.nn.one_hot(inputs, depth)
        elif output_mode == 'multi_hot':
            one_hot_input = backend_module.nn.one_hot(inputs, depth)
            bincounts = backend_module.numpy.where(backend_module.numpy.any(one_hot_input, axis=-2), 1, 0)
    else:
        bincounts = backend_module.numpy.bincount(inputs, minlength=depth)
    bincounts = backend_module.cast(bincounts, dtype)
    return bincounts