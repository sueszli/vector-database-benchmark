"""Perspective Transformer Layer Implementation.

Transform the volume based on 4 x 4 perspective projection matrix.

Reference:
(1) "Perspective Transformer Nets: Perspective Transformer Nets:
Learning Single-View 3D Object Reconstruction without 3D Supervision."
Xinchen Yan, Jimei Yang, Ersin Yumer, Yijie Guo, Honglak Lee. In NIPS 2016
https://papers.nips.cc/paper/6206-perspective-transformer-nets-learning-single-view-3d-object-reconstruction-without-3d-supervision.pdf

(2) Official implementation in Torch: https://github.com/xcyan/ptnbhwd

(3) 2D Transformer implementation in TF:
github.com/tensorflow/models/tree/master/research/transformer

"""
import tensorflow as tf

def transformer(voxels, theta, out_size, z_near, z_far, name='PerspectiveTransformer'):
    if False:
        i = 10
        return i + 15
    'Perspective Transformer Layer.\n\n  Args:\n    voxels: A tensor of size [num_batch, depth, height, width, num_channels].\n      It is the output of a deconv/upsampling conv network (tf.float32).\n    theta: A tensor of size [num_batch, 16].\n      It is the inverse camera transformation matrix (tf.float32).\n    out_size: A tuple representing the size of output of\n      transformer layer (float).\n    z_near: A number representing the near clipping plane (float).\n    z_far: A number representing the far clipping plane (float).\n\n  Returns:\n    A transformed tensor (tf.float32).\n\n  '

    def _repeat(x, n_repeats):
        if False:
            while True:
                i = 10
        with tf.variable_scope('_repeat'):
            rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.stack([n_repeats])), 1), [1, 0])
            rep = tf.to_int32(rep)
            x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
            return tf.reshape(x, [-1])

    def _interpolate(im, x, y, z, out_size):
        if False:
            return 10
        'Bilinear interploation layer.\n\n    Args:\n      im: A 5D tensor of size [num_batch, depth, height, width, num_channels].\n        It is the input volume for the transformation layer (tf.float32).\n      x: A tensor of size [num_batch, out_depth, out_height, out_width]\n        representing the inverse coordinate mapping for x (tf.float32).\n      y: A tensor of size [num_batch, out_depth, out_height, out_width]\n        representing the inverse coordinate mapping for y (tf.float32).\n      z: A tensor of size [num_batch, out_depth, out_height, out_width]\n        representing the inverse coordinate mapping for z (tf.float32).\n      out_size: A tuple representing the output size of transformation layer\n        (float).\n\n    Returns:\n      A transformed tensor (tf.float32).\n\n    '
        with tf.variable_scope('_interpolate'):
            num_batch = im.get_shape().as_list()[0]
            depth = im.get_shape().as_list()[1]
            height = im.get_shape().as_list()[2]
            width = im.get_shape().as_list()[3]
            channels = im.get_shape().as_list()[4]
            x = tf.to_float(x)
            y = tf.to_float(y)
            z = tf.to_float(z)
            depth_f = tf.to_float(depth)
            height_f = tf.to_float(height)
            width_f = tf.to_float(width)
            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]
            zero = tf.zeros([], dtype='int32')
            max_z = tf.to_int32(tf.shape(im)[1] - 1)
            max_y = tf.to_int32(tf.shape(im)[2] - 1)
            max_x = tf.to_int32(tf.shape(im)[3] - 1)
            x = (x + 1.0) * width_f / 2.0
            y = (y + 1.0) * height_f / 2.0
            z = (z + 1.0) * depth_f / 2.0
            x0 = tf.to_int32(tf.floor(x))
            x1 = x0 + 1
            y0 = tf.to_int32(tf.floor(y))
            y1 = y0 + 1
            z0 = tf.to_int32(tf.floor(z))
            z1 = z0 + 1
            x0_clip = tf.clip_by_value(x0, zero, max_x)
            x1_clip = tf.clip_by_value(x1, zero, max_x)
            y0_clip = tf.clip_by_value(y0, zero, max_y)
            y1_clip = tf.clip_by_value(y1, zero, max_y)
            z0_clip = tf.clip_by_value(z0, zero, max_z)
            z1_clip = tf.clip_by_value(z1, zero, max_z)
            dim3 = width
            dim2 = width * height
            dim1 = width * height * depth
            base = _repeat(tf.range(num_batch) * dim1, out_depth * out_height * out_width)
            base_z0_y0 = base + z0_clip * dim2 + y0_clip * dim3
            base_z0_y1 = base + z0_clip * dim2 + y1_clip * dim3
            base_z1_y0 = base + z1_clip * dim2 + y0_clip * dim3
            base_z1_y1 = base + z1_clip * dim2 + y1_clip * dim3
            idx_z0_y0_x0 = base_z0_y0 + x0_clip
            idx_z0_y0_x1 = base_z0_y0 + x1_clip
            idx_z0_y1_x0 = base_z0_y1 + x0_clip
            idx_z0_y1_x1 = base_z0_y1 + x1_clip
            idx_z1_y0_x0 = base_z1_y0 + x0_clip
            idx_z1_y0_x1 = base_z1_y0 + x1_clip
            idx_z1_y1_x0 = base_z1_y1 + x0_clip
            idx_z1_y1_x1 = base_z1_y1 + x1_clip
            im_flat = tf.reshape(im, tf.stack([-1, channels]))
            im_flat = tf.to_float(im_flat)
            i_z0_y0_x0 = tf.gather(im_flat, idx_z0_y0_x0)
            i_z0_y0_x1 = tf.gather(im_flat, idx_z0_y0_x1)
            i_z0_y1_x0 = tf.gather(im_flat, idx_z0_y1_x0)
            i_z0_y1_x1 = tf.gather(im_flat, idx_z0_y1_x1)
            i_z1_y0_x0 = tf.gather(im_flat, idx_z1_y0_x0)
            i_z1_y0_x1 = tf.gather(im_flat, idx_z1_y0_x1)
            i_z1_y1_x0 = tf.gather(im_flat, idx_z1_y1_x0)
            i_z1_y1_x1 = tf.gather(im_flat, idx_z1_y1_x1)
            x0_f = tf.to_float(x0)
            x1_f = tf.to_float(x1)
            y0_f = tf.to_float(y0)
            y1_f = tf.to_float(y1)
            z0_f = tf.to_float(z0)
            z1_f = tf.to_float(z1)
            x0_valid = tf.to_float(tf.less_equal(x0, max_x) & tf.greater_equal(x0, 0))
            x1_valid = tf.to_float(tf.less_equal(x1, max_x) & tf.greater_equal(x1, 0))
            y0_valid = tf.to_float(tf.less_equal(y0, max_y) & tf.greater_equal(y0, 0))
            y1_valid = tf.to_float(tf.less_equal(y1, max_y) & tf.greater_equal(y1, 0))
            z0_valid = tf.to_float(tf.less_equal(z0, max_z) & tf.greater_equal(z0, 0))
            z1_valid = tf.to_float(tf.less_equal(z1, max_z) & tf.greater_equal(z1, 0))
            w_z0_y0_x0 = tf.expand_dims((x1_f - x) * (y1_f - y) * (z1_f - z) * x1_valid * y1_valid * z1_valid, 1)
            w_z0_y0_x1 = tf.expand_dims((x - x0_f) * (y1_f - y) * (z1_f - z) * x0_valid * y1_valid * z1_valid, 1)
            w_z0_y1_x0 = tf.expand_dims((x1_f - x) * (y - y0_f) * (z1_f - z) * x1_valid * y0_valid * z1_valid, 1)
            w_z0_y1_x1 = tf.expand_dims((x - x0_f) * (y - y0_f) * (z1_f - z) * x0_valid * y0_valid * z1_valid, 1)
            w_z1_y0_x0 = tf.expand_dims((x1_f - x) * (y1_f - y) * (z - z0_f) * x1_valid * y1_valid * z0_valid, 1)
            w_z1_y0_x1 = tf.expand_dims((x - x0_f) * (y1_f - y) * (z - z0_f) * x0_valid * y1_valid * z0_valid, 1)
            w_z1_y1_x0 = tf.expand_dims((x1_f - x) * (y - y0_f) * (z - z0_f) * x1_valid * y0_valid * z0_valid, 1)
            w_z1_y1_x1 = tf.expand_dims((x - x0_f) * (y - y0_f) * (z - z0_f) * x0_valid * y0_valid * z0_valid, 1)
            output = tf.add_n([w_z0_y0_x0 * i_z0_y0_x0, w_z0_y0_x1 * i_z0_y0_x1, w_z0_y1_x0 * i_z0_y1_x0, w_z0_y1_x1 * i_z0_y1_x1, w_z1_y0_x0 * i_z1_y0_x0, w_z1_y0_x1 * i_z1_y0_x1, w_z1_y1_x0 * i_z1_y1_x0, w_z1_y1_x1 * i_z1_y1_x1])
            return output

    def _meshgrid(depth, height, width, z_near, z_far):
        if False:
            return 10
        with tf.variable_scope('_meshgrid'):
            x_t = tf.reshape(tf.tile(tf.linspace(-1.0, 1.0, width), [height * depth]), [depth, height, width])
            y_t = tf.reshape(tf.tile(tf.linspace(-1.0, 1.0, height), [width * depth]), [depth, width, height])
            y_t = tf.transpose(y_t, [0, 2, 1])
            sample_grid = tf.tile(tf.linspace(float(z_near), float(z_far), depth), [width * height])
            z_t = tf.reshape(sample_grid, [height, width, depth])
            z_t = tf.transpose(z_t, [2, 0, 1])
            z_t = 1 / z_t
            d_t = 1 / z_t
            x_t /= z_t
            y_t /= z_t
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
            d_t_flat = tf.reshape(d_t, (1, -1))
            ones = tf.ones_like(x_t_flat)
            grid = tf.concat([d_t_flat, y_t_flat, x_t_flat, ones], 0)
            return grid

    def _transform(theta, input_dim, out_size, z_near, z_far):
        if False:
            print('Hello World!')
        with tf.variable_scope('_transform'):
            num_batch = input_dim.get_shape().as_list()[0]
            num_channels = input_dim.get_shape().as_list()[4]
            theta = tf.reshape(theta, (-1, 4, 4))
            theta = tf.cast(theta, 'float32')
            out_depth = out_size[0]
            out_height = out_size[1]
            out_width = out_size[2]
            grid = _meshgrid(out_depth, out_height, out_width, z_near, z_far)
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            grid = tf.tile(grid, tf.stack([num_batch]))
            grid = tf.reshape(grid, tf.stack([num_batch, 4, -1]))
            t_g = tf.matmul(theta, grid)
            z_s = tf.slice(t_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(t_g, [0, 1, 0], [-1, 1, -1])
            x_s = tf.slice(t_g, [0, 2, 0], [-1, 1, -1])
            z_s_flat = tf.reshape(z_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            x_s_flat = tf.reshape(x_s, [-1])
            input_transformed = _interpolate(input_dim, x_s_flat, y_s_flat, z_s_flat, out_size)
            output = tf.reshape(input_transformed, tf.stack([num_batch, out_depth, out_height, out_width, num_channels]))
            return output
    with tf.variable_scope(name):
        output = _transform(theta, voxels, out_size, z_near, z_far)
        return output