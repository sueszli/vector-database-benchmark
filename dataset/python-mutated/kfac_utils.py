import tensorflow as tf

def gmatmul(tensor_a, tensor_b, transpose_a=False, transpose_b=False, reduce_dim=None):
    if False:
        while True:
            i = 10
    "\n    Do a matrix multiplication with tensor 'a' and 'b', even when their shape do not match\n\n    :param tensor_a: (TensorFlow Tensor)\n    :param tensor_b: (TensorFlow Tensor)\n    :param transpose_a: (bool) If 'a' needs transposing\n    :param transpose_b: (bool) If 'b' needs transposing\n    :param reduce_dim: (int) the multiplication over the dim\n    :return: (TensorFlow Tensor) a * b\n    "
    assert reduce_dim is not None
    if len(tensor_a.get_shape()) == 2 and len(tensor_b.get_shape()) > 2:
        b_shape = tensor_b.get_shape()
        if reduce_dim != 0:
            b_dims = list(range(len(b_shape)))
            b_dims.remove(reduce_dim)
            b_dims.insert(0, reduce_dim)
            tensor_b = tf.transpose(tensor_b, b_dims)
        b_t_shape = tensor_b.get_shape()
        tensor_b = tf.reshape(tensor_b, [int(b_shape[reduce_dim]), -1])
        result = tf.matmul(tensor_a, tensor_b, transpose_a=transpose_a, transpose_b=transpose_b)
        result = tf.reshape(result, b_t_shape)
        if reduce_dim != 0:
            b_dims = list(range(len(b_shape)))
            b_dims.remove(0)
            b_dims.insert(reduce_dim, 0)
            result = tf.transpose(result, b_dims)
        return result
    elif len(tensor_a.get_shape()) > 2 and len(tensor_b.get_shape()) == 2:
        a_shape = tensor_a.get_shape()
        outter_dim = len(a_shape) - 1
        reduce_dim = len(a_shape) - reduce_dim - 1
        if reduce_dim != outter_dim:
            a_dims = list(range(len(a_shape)))
            a_dims.remove(reduce_dim)
            a_dims.insert(outter_dim, reduce_dim)
            tensor_a = tf.transpose(tensor_a, a_dims)
        a_t_shape = tensor_a.get_shape()
        tensor_a = tf.reshape(tensor_a, [-1, int(a_shape[reduce_dim])])
        result = tf.matmul(tensor_a, tensor_b, transpose_a=transpose_a, transpose_b=transpose_b)
        result = tf.reshape(result, a_t_shape)
        if reduce_dim != outter_dim:
            a_dims = list(range(len(a_shape)))
            a_dims.remove(outter_dim)
            a_dims.insert(reduce_dim, outter_dim)
            result = tf.transpose(result, a_dims)
        return result
    elif len(tensor_a.get_shape()) == 2 and len(tensor_b.get_shape()) == 2:
        return tf.matmul(tensor_a, tensor_b, transpose_a=transpose_a, transpose_b=transpose_b)
    assert False, 'something went wrong'

def clipout_neg(vec, threshold=1e-06):
    if False:
        for i in range(10):
            print('nop')
    '\n    clip to 0 if input lower than threshold value\n\n    :param vec: (TensorFlow Tensor)\n    :param threshold: (float) the cutoff threshold\n    :return: (TensorFlow Tensor) clipped input\n    '
    mask = tf.cast(vec > threshold, tf.float32)
    return mask * vec

def detect_min_val(input_mat, var, threshold=1e-06, name='', debug=False):
    if False:
        return 10
    '\n    If debug is not set, will run clipout_neg. Else, will clip and print out odd eigen values\n\n    :param input_mat: (TensorFlow Tensor)\n    :param var: (TensorFlow Tensor) variable\n    :param threshold: (float) the cutoff threshold\n    :param name: (str) the name of the variable\n    :param debug: (bool) debug function\n    :return: (TensorFlow Tensor) clipped tensor\n    '
    eigen_min = tf.reduce_min(input_mat)
    eigen_max = tf.reduce_max(input_mat)
    eigen_ratio = eigen_max / eigen_min
    input_mat_clipped = clipout_neg(input_mat, threshold)
    if debug:
        input_mat_clipped = tf.cond(tf.logical_or(tf.greater(eigen_ratio, 0.0), tf.less(eigen_ratio, -500)), lambda : input_mat_clipped, lambda : tf.Print(input_mat_clipped, [tf.convert_to_tensor('odd ratio ' + name + ' eigen values!!!'), tf.convert_to_tensor(var.name), eigen_min, eigen_max, eigen_ratio]))
    return input_mat_clipped

def factor_reshape(eigen_vectors, eigen_values, grad, fac_idx=0, f_type='act'):
    if False:
        while True:
            i = 10
    '\n    factor and reshape input eigen values\n\n    :param eigen_vectors: ([TensorFlow Tensor]) eigen vectors\n    :param eigen_values: ([TensorFlow Tensor]) eigen values\n    :param grad: ([TensorFlow Tensor]) gradient\n    :param fac_idx: (int) index that should be factored\n    :param f_type: (str) function type to factor and reshape\n    :return: ([TensorFlow Tensor], [TensorFlow Tensor]) factored and reshaped eigen vectors\n            and eigen values\n    '
    grad_shape = grad.get_shape()
    if f_type == 'act':
        assert eigen_values.get_shape()[0] == grad_shape[fac_idx]
        expanded_shape = [1] * len(grad_shape)
        expanded_shape[fac_idx] = -1
        eigen_values = tf.reshape(eigen_values, expanded_shape)
    if f_type == 'grad':
        assert eigen_values.get_shape()[0] == grad_shape[len(grad_shape) - fac_idx - 1]
        expanded_shape = [1] * len(grad_shape)
        expanded_shape[len(grad_shape) - fac_idx - 1] = -1
        eigen_values = tf.reshape(eigen_values, expanded_shape)
    return (eigen_vectors, eigen_values)