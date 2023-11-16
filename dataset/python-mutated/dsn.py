"""Functions to create a DSN model and add the different losses to it.

Specifically, in this file we define the:
  - Shared Encoding Similarity Loss Module, with:
    - The MMD Similarity method
    - The Correlation Similarity method
    - The Gradient Reversal (Domain-Adversarial) method
  - Difference Loss Module
  - Reconstruction Loss Module
  - Task Loss Module
"""
from functools import partial
import tensorflow as tf
import losses
import models
import utils
slim = tf.contrib.slim

def dsn_loss_coefficient(params):
    if False:
        while True:
            i = 10
    "The global_step-dependent weight that specifies when to kick in DSN losses.\n\n  Args:\n    params: A dictionary of parameters. Expecting 'domain_separation_startpoint'\n\n  Returns:\n    A weight to that effectively enables or disables the DSN-related losses,\n    i.e. similarity, difference, and reconstruction losses.\n  "
    return tf.where(tf.less(slim.get_or_create_global_step(), params['domain_separation_startpoint']), 1e-10, 1.0)

def create_model(source_images, source_labels, domain_selection_mask, target_images, target_labels, similarity_loss, params, basic_tower_name):
    if False:
        while True:
            i = 10
    "Creates a DSN model.\n\n  Args:\n    source_images: images from the source domain, a tensor of size\n      [batch_size, height, width, channels]\n    source_labels: a dictionary with the name, tensor pairs. 'classes' is one-\n      hot for the number of classes.\n    domain_selection_mask: a boolean tensor of size [batch_size, ] which denotes\n      the labeled images that belong to the source domain.\n    target_images: images from the target domain, a tensor of size\n      [batch_size, height width, channels].\n    target_labels: a dictionary with the name, tensor pairs.\n    similarity_loss: The type of method to use for encouraging\n      the codes from the shared encoder to be similar.\n    params: A dictionary of parameters. Expecting 'weight_decay',\n      'layers_to_regularize', 'use_separation', 'domain_separation_startpoint',\n      'alpha_weight', 'beta_weight', 'gamma_weight', 'recon_loss_name',\n      'decoder_name', 'encoder_name'\n    basic_tower_name: the name of the tower to use for the shared encoder.\n\n  Raises:\n    ValueError: if the arch is not one of the available architectures.\n  "
    network = getattr(models, basic_tower_name)
    num_classes = source_labels['classes'].get_shape().as_list()[1]
    network = partial(network, num_classes=num_classes)
    source_endpoints = add_task_loss(source_images, source_labels, network, params)
    if similarity_loss == 'none':
        return
    with tf.variable_scope('towers', reuse=True):
        (target_logits, target_endpoints) = network(target_images, weight_decay=params['weight_decay'], prefix='target')
    target_accuracy = utils.accuracy(tf.argmax(target_logits, 1), tf.argmax(target_labels['classes'], 1))
    if 'quaternions' in target_labels:
        target_quaternion_loss = losses.log_quaternion_loss(target_labels['quaternions'], target_endpoints['quaternion_pred'], params)
        tf.summary.scalar('eval/Target quaternions', target_quaternion_loss)
    tf.summary.scalar('eval/Target accuracy', target_accuracy)
    source_shared = source_endpoints[params['layers_to_regularize']]
    target_shared = target_endpoints[params['layers_to_regularize']]
    indices = tf.range(0, source_shared.get_shape().as_list()[0])
    indices = tf.boolean_mask(indices, domain_selection_mask)
    add_similarity_loss(similarity_loss, tf.gather(source_shared, indices), tf.gather(target_shared, indices), params)
    if params['use_separation']:
        add_autoencoders(source_images, source_shared, target_images, target_shared, params=params)

def add_similarity_loss(method_name, source_samples, target_samples, params, scope=None):
    if False:
        i = 10
        return i + 15
    "Adds a loss encouraging the shared encoding from each domain to be similar.\n\n  Args:\n    method_name: the name of the encoding similarity method to use. Valid\n      options include `dann_loss', `mmd_loss' or `correlation_loss'.\n    source_samples: a tensor of shape [num_samples, num_features].\n    target_samples: a tensor of shape [num_samples, num_features].\n    params: a dictionary of parameters. Expecting 'gamma_weight'.\n    scope: optional name scope for summary tags.\n  Raises:\n    ValueError: if `method_name` is not recognized.\n  "
    weight = dsn_loss_coefficient(params) * params['gamma_weight']
    method = getattr(losses, method_name)
    method(source_samples, target_samples, weight, scope)

def add_reconstruction_loss(recon_loss_name, images, recons, weight, domain):
    if False:
        for i in range(10):
            print('nop')
    'Adds a reconstruction loss.\n\n  Args:\n    recon_loss_name: The name of the reconstruction loss.\n    images: A `Tensor` of size [batch_size, height, width, 3].\n    recons: A `Tensor` whose size matches `images`.\n    weight: A scalar coefficient for the loss.\n    domain: The name of the domain being reconstructed.\n\n  Raises:\n    ValueError: If `recon_loss_name` is not recognized.\n  '
    if recon_loss_name == 'sum_of_pairwise_squares':
        loss_fn = tf.contrib.losses.mean_pairwise_squared_error
    elif recon_loss_name == 'sum_of_squares':
        loss_fn = tf.contrib.losses.mean_squared_error
    else:
        raise ValueError('recon_loss_name value [%s] not recognized.' % recon_loss_name)
    loss = loss_fn(recons, images, weight)
    assert_op = tf.Assert(tf.is_finite(loss), [loss])
    with tf.control_dependencies([assert_op]):
        tf.summary.scalar('losses/%s Recon Loss' % domain, loss)

def add_autoencoders(source_data, source_shared, target_data, target_shared, params):
    if False:
        i = 10
        return i + 15
    "Adds the encoders/decoders for our domain separation model w/ incoherence.\n\n  Args:\n    source_data: images from the source domain, a tensor of size\n      [batch_size, height, width, channels]\n    source_shared: a tensor with first dimension batch_size\n    target_data: images from the target domain, a tensor of size\n      [batch_size, height, width, channels]\n    target_shared: a tensor with first dimension batch_size\n    params: A dictionary of parameters. Expecting 'layers_to_regularize',\n      'beta_weight', 'alpha_weight', 'recon_loss_name', 'decoder_name',\n      'encoder_name', 'weight_decay'\n  "

    def normalize_images(images):
        if False:
            i = 10
            return i + 15
        images -= tf.reduce_min(images)
        return images / tf.reduce_max(images)

    def concat_operation(shared_repr, private_repr):
        if False:
            return 10
        return shared_repr + private_repr
    mu = dsn_loss_coefficient(params)
    concat_layer = params['layers_to_regularize']
    difference_loss_weight = params['beta_weight'] * mu
    recon_loss_weight = params['alpha_weight'] * mu
    recon_loss_name = params['recon_loss_name']
    decoder_name = params['decoder_name']
    encoder_name = params['encoder_name']
    (_, height, width, _) = source_data.get_shape().as_list()
    code_size = source_shared.get_shape().as_list()[-1]
    weight_decay = params['weight_decay']
    encoder_fn = getattr(models, encoder_name)
    with tf.variable_scope('source_encoder'):
        source_endpoints = encoder_fn(source_data, code_size, weight_decay=weight_decay)
    with tf.variable_scope('target_encoder'):
        target_endpoints = encoder_fn(target_data, code_size, weight_decay=weight_decay)
    decoder_fn = getattr(models, decoder_name)
    decoder = partial(decoder_fn, height=height, width=width, channels=source_data.get_shape().as_list()[-1], weight_decay=weight_decay)
    source_private = source_endpoints[concat_layer]
    target_private = target_endpoints[concat_layer]
    with tf.variable_scope('decoder'):
        source_recons = decoder(concat_operation(source_shared, source_private))
    with tf.variable_scope('decoder', reuse=True):
        source_private_recons = decoder(concat_operation(tf.zeros_like(source_private), source_private))
        source_shared_recons = decoder(concat_operation(source_shared, tf.zeros_like(source_shared)))
    with tf.variable_scope('decoder', reuse=True):
        target_recons = decoder(concat_operation(target_shared, target_private))
        target_shared_recons = decoder(concat_operation(target_shared, tf.zeros_like(target_shared)))
        target_private_recons = decoder(concat_operation(tf.zeros_like(target_private), target_private))
    losses.difference_loss(source_private, source_shared, weight=difference_loss_weight, name='Source')
    losses.difference_loss(target_private, target_shared, weight=difference_loss_weight, name='Target')
    add_reconstruction_loss(recon_loss_name, source_data, source_recons, recon_loss_weight, 'source')
    add_reconstruction_loss(recon_loss_name, target_data, target_recons, recon_loss_weight, 'target')
    source_reconstructions = tf.concat(axis=2, values=map(normalize_images, [source_data, source_recons, source_shared_recons, source_private_recons]))
    target_reconstructions = tf.concat(axis=2, values=map(normalize_images, [target_data, target_recons, target_shared_recons, target_private_recons]))
    tf.summary.image('Source Images:Recons:RGB', source_reconstructions[:, :, :, :3], max_outputs=10)
    tf.summary.image('Target Images:Recons:RGB', target_reconstructions[:, :, :, :3], max_outputs=10)
    if source_reconstructions.get_shape().as_list()[3] == 4:
        tf.summary.image('Source Images:Recons:Depth', source_reconstructions[:, :, :, 3:4], max_outputs=10)
        tf.summary.image('Target Images:Recons:Depth', target_reconstructions[:, :, :, 3:4], max_outputs=10)

def add_task_loss(source_images, source_labels, basic_tower, params):
    if False:
        for i in range(10):
            print('nop')
    "Adds a classification and/or pose estimation loss to the model.\n\n  Args:\n    source_images: images from the source domain, a tensor of size\n      [batch_size, height, width, channels]\n    source_labels: labels from the source domain, a tensor of size [batch_size].\n      or a tuple of (quaternions, class_labels)\n    basic_tower: a function that creates the single tower of the model.\n    params: A dictionary of parameters. Expecting 'weight_decay', 'pose_weight'.\n  Returns:\n    The source endpoints.\n\n  Raises:\n    RuntimeError: if basic tower does not support pose estimation.\n  "
    with tf.variable_scope('towers'):
        (source_logits, source_endpoints) = basic_tower(source_images, weight_decay=params['weight_decay'], prefix='Source')
    if 'quaternions' in source_labels:
        if 'quaternion_pred' not in source_endpoints:
            raise RuntimeError('Please use a model for estimation e.g. pose_mini')
        loss = losses.log_quaternion_loss(source_labels['quaternions'], source_endpoints['quaternion_pred'], params)
        assert_op = tf.Assert(tf.is_finite(loss), [loss])
        with tf.control_dependencies([assert_op]):
            quaternion_loss = loss
            tf.summary.histogram('log_quaternion_loss_hist', quaternion_loss)
        slim.losses.add_loss(quaternion_loss * params['pose_weight'])
        tf.summary.scalar('losses/quaternion_loss', quaternion_loss)
    classification_loss = tf.losses.softmax_cross_entropy(source_labels['classes'], source_logits)
    tf.summary.scalar('losses/classification_loss', classification_loss)
    return source_endpoints