from six.moves import xrange
import tensorflow as tf

def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    if False:
        i = 10
        return i + 15
    'Spatial Transformer Layer\n\n    Implements a spatial transformer layer as described in [1]_.\n    Based on [2]_ and edited by David Dao for Tensorflow.\n\n    Parameters\n    ----------\n    U : float\n        The output of a convolutional net should have the\n        shape [num_batch, height, width, num_channels].\n    theta: float\n        The output of the\n        localisation network should be [num_batch, 6].\n    out_size: tuple of two ints\n        The size of the output of the network (height, width)\n\n    References\n    ----------\n    .. [1]  Spatial Transformer Networks\n            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu\n            Submitted on 5 Jun 2015\n    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py\n\n    Notes\n    -----\n    To initialize the network to the identity transform init\n    ``theta`` to :\n        identity = np.array([[1., 0., 0.],\n                             [0., 1., 0.]])\n        identity = identity.flatten()\n        theta = tf.Variable(initial_value=identity)\n\n    '

    def _repeat(x, n_repeats):
        if False:
            for i in range(10):
                print('nop')
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats])), 1), [1, 0])
            rep = tf.cast(rep, 'int32')
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, out_size):
        if False:
            return 10
        with tf.variable_scope('_interpolate'):
            num_batch = tf.shape(im)[0]
            height = tf.shape(im)[1]
            width = tf.shape(im)[2]
            channels = tf.shape(im)[3]
            x = tf.cast(x, 'float32')
            y = tf.cast(y, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            zero = tf.zeros([], dtype='int32')
            max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
            max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
            x = (x + 1.0) * width_f / 2.0
            y = (y + 1.0) * height_f / 2.0
            x0 = tf.cast(tf.floor(x), 'int32')
            x1 = x0 + 1
            y0 = tf.cast(tf.floor(y), 'int32')
            y1 = y0 + 1
            x0 = tf.clip_by_value(x0, zero, max_x)
            x1 = tf.clip_by_value(x1, zero, max_x)
            y0 = tf.clip_by_value(y0, zero, max_y)
            y1 = tf.clip_by_value(y1, zero, max_y)
            dim2 = width
            dim1 = width * height
            base = _repeat(tf.range(num_batch) * dim1, out_height * out_width)
            base_y0 = base + y0 * dim2
            base_y1 = base + y1 * dim2
            idx_a = base_y0 + x0
            idx_b = base_y1 + x0
            idx_c = base_y0 + x1
            idx_d = base_y1 + x1
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.cast(im_flat, 'float32')
            Ia = tf.gather(im_flat, idx_a)
            Ib = tf.gather(im_flat, idx_b)
            Ic = tf.gather(im_flat, idx_c)
            Id = tf.gather(im_flat, idx_d)
            x0_f = tf.cast(x0, 'float32')
            x1_f = tf.cast(x1, 'float32')
            y0_f = tf.cast(y0, 'float32')
            y1_f = tf.cast(y1, 'float32')
            wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
            wb = tf.expand_dims((x1_f - x) * (y - y0_f), 1)
            wc = tf.expand_dims((x - x0_f) * (y1_f - y), 1)
            wd = tf.expand_dims((x - x0_f) * (y - y0_f), 1)
            output = tf.add_n([wa * Ia, wb * Ib, wc * Ic, wd * Id])
            return output

    def _meshgrid(height, width):
        if False:
            return 10
        with tf.variable_scope('_meshgrid'):
            x_t = tf.matmul(tf.ones(shape=tf.stack([height, 1])), tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1), tf.ones(shape=tf.stack([1, width])))
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(axis=0, values=[x_t_flat, y_t_flat, ones])
            return grid

    def _transform(theta, input_dim, out_size):
        if False:
            return 10
        with tf.variable_scope('_transform'):
            num_batch = tf.shape(input_dim)[0]
            height = tf.shape(input_dim)[1]
            width = tf.shape(input_dim)[2]
            num_channels = tf.shape(input_dim)[3]
            theta = tf.reshape(theta, (-1, 2, 3))
            theta = tf.cast(theta, 'float32')
            height_f = tf.cast(height, 'float32')
            width_f = tf.cast(width, 'float32')
            out_height = out_size[0]
            out_width = out_size[1]
            grid = _meshgrid(out_height, out_width)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 3, -1]))
            T_g = tf.matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, out_size)
            output = tf.reshape(input_transformed, tf.stack([num_batch, out_height, out_width, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(theta, U, out_size)
        return output

def batch_transformer(U, thetas, out_size, name='BatchSpatialTransformer'):
    if False:
        return 10
    'Batch Spatial Transformer Layer\n\n    Parameters\n    ----------\n\n    U : float\n        tensor of inputs [num_batch,height,width,num_channels]\n    thetas : float\n        a set of transformations for each input [num_batch,num_transforms,6]\n    out_size : int\n        the size of the output [out_height,out_width]\n\n    Returns: float\n        Tensor of size [num_batch*num_transforms,out_height,out_width,num_channels]\n    '
    with tf.variable_scope(name):
        (num_batch, num_transforms) = map(int, thetas.get_shape().as_list()[:2])
        indices = [[i] * num_transforms for i in xrange(num_batch)]
        input_repeated = tf.gather(U, tf.reshape(indices, [-1]))
        return transformer(input_repeated, thetas, out_size)