"""Contains the Domain Adaptation via Style Transfer (PixelDA) model components.

A number of details in the implementation make reference to one of the following
works:

- "Unsupervised Representation Learning with Deep Convolutional
    Generative Adversarial Networks""
   https://arxiv.org/abs/1511.06434

This paper makes several architecture recommendations:
1. Use strided convs in discriminator, fractional-strided convs in generator
2. batchnorm everywhere
3. remove fully connected layers for deep models
4. ReLu for all layers in generator, except tanh on output
5. LeakyReLu for everything in discriminator
"""
import functools
import math
import numpy as np
import tensorflow as tf
slim = tf.contrib.slim
from domain_adaptation.pixel_domain_adaptation import pixelda_task_towers

def create_model(hparams, target_images, source_images=None, source_labels=None, is_training=False, noise=None, num_classes=None):
    if False:
        while True:
            i = 10
    'Create a GAN model.\n\n  Arguments:\n    hparams: HParam object specifying model params\n    target_images: A `Tensor` of size [batch_size, height, width, channels]. It\n      is assumed that the images are [-1, 1] normalized.\n    source_images: A `Tensor` of size [batch_size, height, width, channels]. It\n      is assumed that the images are [-1, 1] normalized.\n    source_labels: A `Tensor` of size [batch_size] of categorical labels between\n      [0, num_classes]\n    is_training: whether model is currently training\n    noise: If None, model generates its own noise. Otherwise use provided.\n    num_classes: Number of classes for classification\n\n  Returns:\n    end_points dict with model outputs\n\n  Raises:\n    ValueError: unknown hparams.arch setting\n  '
    if num_classes is None and hparams.arch in ['resnet', 'simple']:
        raise ValueError('Num classes must be provided to create task classifier')
    if target_images.dtype != tf.float32:
        raise ValueError('target_images must be tf.float32 and [-1, 1] normalized.')
    if source_images is not None and source_images.dtype != tf.float32:
        raise ValueError('source_images must be tf.float32 and [-1, 1] normalized.')
    latent_vars = dict()
    if hparams.noise_channel:
        noise_shape = [hparams.batch_size, hparams.noise_dims]
        if noise is not None:
            assert noise.shape.as_list() == noise_shape
            tf.logging.info('Using provided noise')
        else:
            tf.logging.info('Using random noise')
            noise = tf.random_uniform(shape=noise_shape, minval=-1, maxval=1, dtype=tf.float32, name='random_noise')
        latent_vars['noise'] = noise
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose, slim.fully_connected], normalizer_params=batch_norm_params(is_training, hparams.batch_norm_decay), weights_initializer=tf.random_normal_initializer(stddev=hparams.normal_init_std), weights_regularizer=tf.contrib.layers.l2_regularizer(hparams.weight_decay)):
        with slim.arg_scope([slim.conv2d], padding='SAME'):
            if hparams.arch == 'dcgan':
                end_points = dcgan(target_images, latent_vars, hparams, scope='generator')
            elif hparams.arch == 'resnet':
                end_points = resnet_generator(source_images, target_images.shape.as_list()[1:4], hparams=hparams, latent_vars=latent_vars)
            elif hparams.arch == 'residual_interpretation':
                end_points = residual_interpretation_generator(source_images, is_training=is_training, hparams=hparams)
            elif hparams.arch == 'simple':
                end_points = simple_generator(source_images, target_images, is_training=is_training, hparams=hparams, latent_vars=latent_vars)
            elif hparams.arch == 'identity':
                if hparams.generator_steps:
                    raise ValueError('Must set generator_steps=0 for identity arch. Is %s' % hparams.generator_steps)
                transferred_images = source_images
                source_channels = source_images.shape.as_list()[-1]
                target_channels = target_images.shape.as_list()[-1]
                if source_channels == 1 and target_channels == 3:
                    transferred_images = tf.tile(source_images, [1, 1, 1, 3])
                if source_channels == 3 and target_channels == 1:
                    transferred_images = tf.image.rgb_to_grayscale(source_images)
                end_points = {'transferred_images': transferred_images}
            else:
                raise ValueError('Unknown architecture: %s' % hparams.arch)
            if hparams.arch in ['dcgan', 'resnet', 'residual_interpretation', 'simple', 'identity']:
                end_points['transferred_domain_logits'] = predict_domain(end_points['transferred_images'], hparams, is_training=is_training, reuse=False)
                end_points['target_domain_logits'] = predict_domain(target_images, hparams, is_training=is_training, reuse=True)
            if hparams.task_tower != 'none' and hparams.arch in ['resnet', 'residual_interpretation', 'simple', 'identity']:
                with tf.variable_scope('discriminator'):
                    with tf.variable_scope('task_tower'):
                        (end_points['source_task_logits'], end_points['source_quaternion']) = pixelda_task_towers.add_task_specific_model(source_images, hparams, num_classes=num_classes, is_training=is_training, reuse_private=False, private_scope='source_task_classifier', reuse_shared=False)
                        (end_points['transferred_task_logits'], end_points['transferred_quaternion']) = pixelda_task_towers.add_task_specific_model(end_points['transferred_images'], hparams, num_classes=num_classes, is_training=is_training, reuse_private=False, private_scope='transferred_task_classifier', reuse_shared=True)
                        (end_points['target_task_logits'], end_points['target_quaternion']) = pixelda_task_towers.add_task_specific_model(target_images, hparams, num_classes=num_classes, is_training=is_training, reuse_private=True, private_scope='transferred_task_classifier', reuse_shared=True)
    return dict(((k, v) for (k, v) in end_points.iteritems() if v is not None))

def batch_norm_params(is_training, batch_norm_decay):
    if False:
        for i in range(10):
            print('nop')
    return {'is_training': is_training, 'decay': batch_norm_decay, 'epsilon': 0.001}

def lrelu(x, leakiness=0.2):
    if False:
        i = 10
        return i + 15
    'Relu, with optional leaky support.'
    return tf.where(tf.less(x, 0.0), leakiness * x, x, name='leaky_relu')

def upsample(net, num_filters, scale=2, method='resize_conv', scope=None):
    if False:
        return 10
    "Performs spatial upsampling of the given features.\n\n  Args:\n    net: A `Tensor` of shape [batch_size, height, width, filters].\n    num_filters: The number of output filters.\n    scale: The scale of the upsampling. Must be a positive integer greater or\n      equal to two.\n    method: The method by which the features are upsampled. Valid options\n      include 'resize_conv' and 'conv2d_transpose'.\n    scope: An optional variable scope.\n\n  Returns:\n    A new set of features of shape\n      [batch_size, height*scale, width*scale, num_filters].\n\n  Raises:\n    ValueError: if `method` is not valid or\n  "
    if scale < 2:
        raise ValueError('scale must be greater or equal to two.')
    with tf.variable_scope(scope, 'upsample', [net]):
        if method == 'resize_conv':
            net = tf.image.resize_nearest_neighbor(net, [net.shape.as_list()[1] * scale, net.shape.as_list()[2] * scale], align_corners=True, name='resize')
            return slim.conv2d(net, num_filters, stride=1, scope='conv')
        elif method == 'conv2d_transpose':
            return slim.conv2d_transpose(net, num_filters, scope='deconv')
        else:
            raise ValueError('Upsample method [%s] was not recognized.' % method)

def project_latent_vars(hparams, proj_shape, latent_vars, combine_method='sum'):
    if False:
        return 10
    "Generate noise and project to input volume size.\n\n  Args:\n    hparams: The hyperparameter HParams struct.\n    proj_shape: Shape to project noise (not including batch size).\n    latent_vars: dictionary of `'key': Tensor of shape [batch_size, N]`\n    combine_method: How to combine the projected values.\n      sum = project to volume then sum\n      concat = concatenate along last dimension (i.e. channel)\n\n  Returns:\n    If combine_method=sum, a `Tensor` of size `hparams.projection_shape`\n    If combine_method=concat and there are N latent vars, a `Tensor` of size\n      `hparams.projection_shape`, with the last channel multiplied by N\n\n\n  Raises:\n    ValueError: combine_method is not one of sum/concat\n  "
    values = []
    for var in latent_vars:
        with tf.variable_scope(var):
            projected = slim.fully_connected(latent_vars[var], np.prod(proj_shape), activation_fn=tf.nn.relu, normalizer_fn=slim.batch_norm)
            values.append(tf.reshape(projected, [hparams.batch_size] + proj_shape))
    if combine_method == 'sum':
        result = values[0]
        for value in values[1:]:
            result += value
    elif combine_method == 'concat':
        result = tf.concat(values, len(proj_shape))
    else:
        raise ValueError('Unknown combine_method %s' % combine_method)
    tf.logging.info('Latent variables projected to size %s volume', result.shape)
    return result

def resnet_block(net, hparams):
    if False:
        i = 10
        return i + 15
    'Create a resnet block.'
    net_in = net
    net = slim.conv2d(net, hparams.resnet_filters, stride=1, normalizer_fn=slim.batch_norm, activation_fn=tf.nn.relu)
    net = slim.conv2d(net, hparams.resnet_filters, stride=1, normalizer_fn=slim.batch_norm, activation_fn=None)
    if hparams.resnet_residuals:
        net += net_in
    return net

def resnet_stack(images, output_shape, hparams, scope=None):
    if False:
        print('Hello World!')
    'Create a resnet style transfer block.\n\n  Args:\n    images: [batch-size, height, width, channels] image tensor to feed as input\n    output_shape: output image shape in form [height, width, channels]\n    hparams: hparams objects\n    scope: Variable scope\n\n  Returns:\n    Images after processing with resnet blocks.\n  '
    end_points = {}
    if hparams.noise_channel:
        end_points['noise'] = images[:, :, :, -1]
    assert images.shape.as_list()[1:3] == output_shape[0:2]
    with tf.variable_scope(scope, 'resnet_style_transfer', [images]):
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, kernel_size=[hparams.generator_kernel_size] * 2, stride=1):
            net = slim.conv2d(images, hparams.resnet_filters, normalizer_fn=None, activation_fn=tf.nn.relu)
            for block in range(hparams.resnet_blocks):
                net = resnet_block(net, hparams)
                end_points['resnet_block_{}'.format(block)] = net
            net = slim.conv2d(net, output_shape[-1], kernel_size=[1, 1], normalizer_fn=None, activation_fn=tf.nn.tanh, scope='conv_out')
            end_points['transferred_images'] = net
        return (net, end_points)

def predict_domain(images, hparams, is_training=False, reuse=False, scope='discriminator'):
    if False:
        print('Hello World!')
    "Creates a discriminator for a GAN.\n\n  Args:\n    images: A `Tensor` of size [batch_size, height, width, channels]. It is\n      assumed that the images are centered between -1 and 1.\n    hparams: hparam object with params for discriminator\n    is_training: Specifies whether or not we're training or testing.\n    reuse: Whether to reuse variable scope\n    scope: An optional variable_scope.\n\n  Returns:\n    [batch size, 1] - logit output of discriminator.\n  "
    with tf.variable_scope(scope, 'discriminator', [images], reuse=reuse):
        lrelu_partial = functools.partial(lrelu, leakiness=hparams.lrelu_leakiness)
        with slim.arg_scope([slim.conv2d], kernel_size=[hparams.discriminator_kernel_size] * 2, activation_fn=lrelu_partial, stride=2, normalizer_fn=slim.batch_norm):

            def add_noise(hidden, scope_num=None):
                if False:
                    while True:
                        i = 10
                if scope_num:
                    hidden = slim.dropout(hidden, hparams.discriminator_dropout_keep_prob, is_training=is_training, scope='dropout_%s' % scope_num)
                if hparams.discriminator_noise_stddev == 0:
                    return hidden
                return hidden + tf.random_normal(hidden.shape.as_list(), mean=0.0, stddev=hparams.discriminator_noise_stddev)
            if hparams.discriminator_image_noise:
                images = add_noise(images)
            net = slim.conv2d(images, hparams.num_discriminator_filters, normalizer_fn=None, stride=hparams.discriminator_first_stride, scope='conv1_stride%s' % hparams.discriminator_first_stride)
            net = add_noise(net, 1)
            block_id = 2
            while net.shape.as_list()[1] > hparams.projection_shape_size:
                num_filters = int(hparams.num_discriminator_filters * hparams.discriminator_filter_factor ** (block_id - 1))
                for conv_id in range(1, hparams.discriminator_conv_block_size):
                    net = slim.conv2d(net, num_filters, stride=1, scope='conv_%s_%s' % (block_id, conv_id))
                if hparams.discriminator_do_pooling:
                    net = slim.conv2d(net, num_filters, scope='conv_%s_prepool' % block_id)
                    net = slim.avg_pool2d(net, kernel_size=[2, 2], stride=2, scope='pool_%s' % block_id)
                else:
                    net = slim.conv2d(net, num_filters, scope='conv_%s_stride2' % block_id)
                net = add_noise(net, block_id)
                block_id += 1
            net = slim.flatten(net)
            net = slim.fully_connected(net, 1, normalizer_fn=None, activation_fn=None, scope='fc_logit_out')
    return net

def dcgan_generator(images, output_shape, hparams, scope=None):
    if False:
        i = 10
        return i + 15
    "Transforms the visual style of the input images.\n\n  Args:\n    images: A `Tensor` of shape [batch_size, height, width, channels].\n    output_shape: A list or tuple of 3 elements: the output height, width and\n      number of channels.\n    hparams: hparams object with generator parameters\n    scope: Scope to place generator inside\n\n  Returns:\n    A `Tensor` of shape [batch_size, height, width, output_channels] which\n    represents the result of style transfer.\n\n  Raises:\n    ValueError: If `output_shape` is not a list or tuple or if it doesn't have\n    three elements or if `output_shape` or `images` arent square.\n  "
    if not isinstance(output_shape, (tuple, list)):
        raise ValueError('output_shape must be a tuple or list.')
    elif len(output_shape) != 3:
        raise ValueError('output_shape must have three elements.')
    if output_shape[0] != output_shape[1]:
        raise ValueError('output_shape must be square')
    if images.shape.as_list()[1] != images.shape.as_list()[2]:
        raise ValueError('images height and width must match.')
    outdim = output_shape[0]
    indim = images.shape.as_list()[1]
    num_iterations = int(math.ceil(math.log(float(outdim) / float(indim), 2.0)))
    with slim.arg_scope([slim.conv2d, slim.conv2d_transpose], kernel_size=[hparams.generator_kernel_size] * 2, stride=2):
        with tf.variable_scope(scope or 'generator'):
            net = images
            for i in range(num_iterations):
                num_filters = hparams.num_decoder_filters * 2 ** (num_iterations - i - 1)
                net = slim.conv2d_transpose(net, num_filters, scope='deconv_%s' % i)
            dif = net.shape.as_list()[1] - outdim
            low = dif / 2
            high = net.shape.as_list()[1] - low
            net = net[:, low:high, low:high, :]
            net = slim.conv2d(net, output_shape[2], kernel_size=[1, 1], stride=1, normalizer_fn=None, activation_fn=tf.tanh, scope='conv_out')
    return net

def dcgan(target_images, latent_vars, hparams, scope='dcgan'):
    if False:
        return 10
    "Creates the PixelDA model.\n\n  Args:\n    target_images: A `Tensor` of shape [batch_size, height, width, 3]\n      sampled from the image domain to which we want to transfer.\n    latent_vars: dictionary of 'key': Tensor of shape [batch_size, N]\n    hparams: The hyperparameter map.\n    scope: Surround generator component with this scope\n\n  Returns:\n    A dictionary of model outputs.\n  "
    proj_shape = [hparams.projection_shape_size, hparams.projection_shape_size, hparams.projection_shape_channels]
    source_volume = project_latent_vars(hparams, proj_shape, latent_vars, combine_method='concat')
    with tf.variable_scope(scope, 'generator', [target_images]):
        transferred_images = dcgan_generator(source_volume, output_shape=target_images.shape.as_list()[1:4], hparams=hparams)
        assert transferred_images.shape.as_list() == target_images.shape.as_list()
    return {'transferred_images': transferred_images}

def resnet_generator(images, output_shape, hparams, latent_vars=None):
    if False:
        return 10
    "Creates a ResNet-based generator.\n\n  Args:\n    images: A `Tensor` of shape [batch_size, height, width, num_channels]\n      sampled from the image domain from which we want to transfer\n    output_shape: A length-3 array indicating the height, width and channels of\n      the output.\n    hparams: The hyperparameter map.\n    latent_vars: dictionary of 'key': Tensor of shape [batch_size, N]\n\n  Returns:\n    A dictionary of model outputs.\n  "
    with tf.variable_scope('generator'):
        if latent_vars:
            noise_channel = project_latent_vars(hparams, proj_shape=images.shape.as_list()[1:3] + [1], latent_vars=latent_vars, combine_method='concat')
            images = tf.concat([images, noise_channel], 3)
        (transferred_images, end_points) = resnet_stack(images, output_shape=output_shape, hparams=hparams, scope='resnet_stack')
        end_points['transferred_images'] = transferred_images
    return end_points

def residual_interpretation_block(images, hparams, scope):
    if False:
        i = 10
        return i + 15
    'Learns a residual image which is added to the incoming image.\n\n  Args:\n    images: A `Tensor` of size [batch_size, height, width, 3]\n    hparams: The hyperparameters struct.\n    scope: The name of the variable op scope.\n\n  Returns:\n    The updated images.\n  '
    with tf.variable_scope(scope):
        with slim.arg_scope([slim.conv2d], normalizer_fn=None, kernel_size=[hparams.generator_kernel_size] * 2):
            net = images
            for _ in range(hparams.res_int_convs):
                net = slim.conv2d(net, hparams.res_int_filters, activation_fn=tf.nn.relu)
            net = slim.conv2d(net, 3, activation_fn=tf.nn.tanh)
        images += net
        images = tf.maximum(images, -1.0)
        images = tf.minimum(images, 1.0)
        return images

def residual_interpretation_generator(images, is_training, hparams, latent_vars=None):
    if False:
        i = 10
        return i + 15
    "Creates a generator producing purely residual transformations.\n\n  A residual generator differs from the resnet generator in that each 'block' of\n  the residual generator produces a residual image. Consequently, the 'progress'\n  of the model generation process can be directly observed at inference time,\n  making it easier to diagnose and understand.\n\n  Args:\n    images: A `Tensor` of shape [batch_size, height, width, num_channels]\n      sampled from the image domain from which we want to transfer. It is\n      assumed that the images are centered between -1 and 1.\n    is_training: whether or not the model is training.\n    hparams: The hyperparameter map.\n    latent_vars: dictionary of 'key': Tensor of shape [batch_size, N]\n\n  Returns:\n    A dictionary of model outputs.\n  "
    end_points = {}
    with tf.variable_scope('generator'):
        if latent_vars:
            projected_latent = project_latent_vars(hparams, proj_shape=images.shape.as_list()[1:3] + [images.shape.as_list()[-1]], latent_vars=latent_vars, combine_method='sum')
            images += projected_latent
        with tf.variable_scope(None, 'residual_style_transfer', [images]):
            for i in range(hparams.res_int_blocks):
                images = residual_interpretation_block(images, hparams, 'residual_%d' % i)
                end_points['transferred_images_%d' % i] = images
            end_points['transferred_images'] = images
    return end_points

def simple_generator(source_images, target_images, is_training, hparams, latent_vars):
    if False:
        while True:
            i = 10
    'Simple generator architecture (stack of convs) for trying small models.'
    end_points = {}
    with tf.variable_scope('generator'):
        feed_source_images = source_images
        if latent_vars:
            projected_latent = project_latent_vars(hparams, proj_shape=source_images.shape.as_list()[1:3] + [1], latent_vars=latent_vars, combine_method='concat')
            feed_source_images = tf.concat([source_images, projected_latent], 3)
        end_points = {}
        with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, stride=1, kernel_size=[hparams.generator_kernel_size] * 2):
            net = feed_source_images
            for i in range(1, hparams.simple_num_conv_layers):
                normalizer_fn = None
                if i != 0:
                    normalizer_fn = slim.batch_norm
                net = slim.conv2d(net, hparams.simple_conv_filters, normalizer_fn=normalizer_fn, activation_fn=tf.nn.relu)
            net = slim.conv2d(net, target_images.shape.as_list()[-1], kernel_size=[1, 1], stride=1, normalizer_fn=None, activation_fn=tf.tanh, scope='conv_out')
        transferred_images = net
        assert transferred_images.shape.as_list() == target_images.shape.as_list()
        end_points['transferred_images'] = transferred_images
    return end_points