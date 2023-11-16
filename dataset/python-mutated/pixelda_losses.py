"""Defines the various loss functions in use by the PIXELDA model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
slim = tf.contrib.slim

def add_domain_classifier_losses(end_points, hparams):
    if False:
        i = 10
        return i + 15
    'Adds losses related to the domain-classifier.\n\n  Args:\n    end_points: A map of network end point names to `Tensors`.\n    hparams: The hyperparameters struct.\n\n  Returns:\n    loss: A `Tensor` representing the total task-classifier loss.\n  '
    if hparams.domain_loss_weight == 0:
        tf.logging.info('Domain classifier loss weight is 0, so not creating losses.')
        return 0
    transferred_domain_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.zeros_like(end_points['transferred_domain_logits']), logits=end_points['transferred_domain_logits'])
    tf.summary.scalar('Domain_loss_transferred', transferred_domain_loss)
    target_domain_loss = tf.losses.sigmoid_cross_entropy(multi_class_labels=tf.ones_like(end_points['target_domain_logits']), logits=end_points['target_domain_logits'])
    tf.summary.scalar('Domain_loss_target', target_domain_loss)
    total_domain_loss = transferred_domain_loss + target_domain_loss
    total_domain_loss *= hparams.domain_loss_weight
    tf.summary.scalar('Domain_loss_total', total_domain_loss)
    return total_domain_loss

def log_quaternion_loss_batch(predictions, labels, params):
    if False:
        return 10
    "A helper function to compute the error between quaternions.\n\n  Args:\n    predictions: A Tensor of size [batch_size, 4].\n    labels: A Tensor of size [batch_size, 4].\n    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.\n\n  Returns:\n    A Tensor of size [batch_size], denoting the error between the quaternions.\n  "
    use_logging = params['use_logging']
    assertions = []
    if use_logging:
        assertions.append(tf.Assert(tf.reduce_all(tf.less(tf.abs(tf.reduce_sum(tf.square(predictions), [1]) - 1), 0.0001)), ['The l2 norm of each prediction quaternion vector should be 1.']))
        assertions.append(tf.Assert(tf.reduce_all(tf.less(tf.abs(tf.reduce_sum(tf.square(labels), [1]) - 1), 0.0001)), ['The l2 norm of each label quaternion vector should be 1.']))
    with tf.control_dependencies(assertions):
        product = tf.multiply(predictions, labels)
    internal_dot_products = tf.reduce_sum(product, [1])
    if use_logging:
        internal_dot_products = tf.Print(internal_dot_products, [internal_dot_products, tf.shape(internal_dot_products)], 'internal_dot_products:')
    logcost = tf.log(0.0001 + 1 - tf.abs(internal_dot_products))
    return logcost

def log_quaternion_loss(predictions, labels, params):
    if False:
        while True:
            i = 10
    "A helper function to compute the mean error between batches of quaternions.\n\n  The caller is expected to add the loss to the graph.\n\n  Args:\n    predictions: A Tensor of size [batch_size, 4].\n    labels: A Tensor of size [batch_size, 4].\n    params: A dictionary of parameters. Expecting 'use_logging', 'batch_size'.\n\n  Returns:\n    A Tensor of size 1, denoting the mean error between batches of quaternions.\n  "
    use_logging = params['use_logging']
    logcost = log_quaternion_loss_batch(predictions, labels, params)
    logcost = tf.reduce_sum(logcost, [0])
    batch_size = params['batch_size']
    logcost = tf.multiply(logcost, 1.0 / batch_size, name='log_quaternion_loss')
    if use_logging:
        logcost = tf.Print(logcost, [logcost], '[logcost]', name='log_quaternion_loss_print')
    return logcost

def _quaternion_loss(labels, predictions, weight, batch_size, domain, add_summaries):
    if False:
        for i in range(10):
            print('nop')
    'Creates a Quaternion Loss.\n\n  Args:\n    labels: The true quaternions.\n    predictions: The predicted quaternions.\n    weight: A scalar weight.\n    batch_size: The size of the batches.\n    domain: The name of the domain from which the labels were taken.\n    add_summaries: Whether or not to add summaries for the losses.\n\n  Returns:\n    A `Tensor` representing the loss.\n  '
    assert domain in ['Source', 'Transferred']
    params = {'use_logging': False, 'batch_size': batch_size}
    loss = weight * log_quaternion_loss(labels, predictions, params)
    if add_summaries:
        assert_op = tf.Assert(tf.is_finite(loss), [loss])
        with tf.control_dependencies([assert_op]):
            tf.summary.histogram('Log_Quaternion_Loss_%s' % domain, loss, collections='losses')
            tf.summary.scalar('Task_Quaternion_Loss_%s' % domain, loss, collections='losses')
    return loss

def _add_task_specific_losses(end_points, source_labels, num_classes, hparams, add_summaries=False):
    if False:
        return 10
    'Adds losses related to the task-classifier.\n\n  Args:\n    end_points: A map of network end point names to `Tensors`.\n    source_labels: A dictionary of output labels to `Tensors`.\n    num_classes: The number of classes used by the classifier.\n    hparams: The hyperparameters struct.\n    add_summaries: Whether or not to add the summaries.\n\n  Returns:\n    loss: A `Tensor` representing the total task-classifier loss.\n  '
    one_hot_labels = slim.one_hot_encoding(source_labels['class'], num_classes)
    total_loss = 0
    if 'source_task_logits' in end_points:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=end_points['source_task_logits'], weights=hparams.source_task_loss_weight)
        if add_summaries:
            tf.summary.scalar('Task_Classifier_Loss_Source', loss)
        total_loss += loss
    if 'transferred_task_logits' in end_points:
        loss = tf.losses.softmax_cross_entropy(onehot_labels=one_hot_labels, logits=end_points['transferred_task_logits'], weights=hparams.transferred_task_loss_weight)
        if add_summaries:
            tf.summary.scalar('Task_Classifier_Loss_Transferred', loss)
        total_loss += loss
    if 'quaternion' in source_labels:
        total_loss += _quaternion_loss(source_labels['quaternion'], end_points['source_quaternion'], hparams.source_pose_weight, hparams.batch_size, 'Source', add_summaries)
        total_loss += _quaternion_loss(source_labels['quaternion'], end_points['transferred_quaternion'], hparams.transferred_pose_weight, hparams.batch_size, 'Transferred', add_summaries)
    if add_summaries:
        tf.summary.scalar('Task_Loss_Total', total_loss)
    return total_loss

def _transferred_similarity_loss(reconstructions, source_images, weight=1.0, method='mse', max_diff=0.4, name='similarity'):
    if False:
        return 10
    'Computes a loss encouraging similarity between source and transferred.\n\n  Args:\n    reconstructions: A `Tensor` of shape [batch_size, height, width, channels]\n    source_images: A `Tensor` of shape [batch_size, height, width, channels].\n    weight: Multiple similarity loss by this weight before returning\n    method: One of:\n      mpse = Mean Pairwise Squared Error\n      mse = Mean Squared Error\n      hinged_mse = Computes the mean squared error using squared differences\n        greater than hparams.transferred_similarity_max_diff\n      hinged_mae = Computes the mean absolute error using absolute\n        differences greater than hparams.transferred_similarity_max_diff.\n    max_diff: Maximum unpenalized difference for hinged losses\n    name: Identifying name to use for creating summaries\n\n\n  Returns:\n    A `Tensor` representing the transferred similarity loss.\n\n  Raises:\n    ValueError: if `method` is not recognized.\n  '
    if weight == 0:
        return 0
    source_channels = source_images.shape.as_list()[-1]
    reconstruction_channels = reconstructions.shape.as_list()[-1]
    if source_channels == 1 and reconstruction_channels != 1:
        source_images = tf.tile(source_images, [1, 1, 1, reconstruction_channels])
    if reconstruction_channels == 1 and source_channels != 1:
        reconstructions = tf.tile(reconstructions, [1, 1, 1, source_channels])
    if method == 'mpse':
        reconstruction_similarity_loss_fn = tf.contrib.losses.mean_pairwise_squared_error
    elif method == 'masked_mpse':

        def masked_mpse(predictions, labels, weight):
            if False:
                return 10
            'Masked mpse assuming we have a depth to create a mask from.'
            assert labels.shape.as_list()[-1] == 4
            mask = tf.to_float(tf.less(labels[:, :, :, 3:4], 0.99))
            mask = tf.tile(mask, [1, 1, 1, 4])
            predictions *= mask
            labels *= mask
            tf.image_summary('masked_pred', predictions)
            tf.image_summary('masked_label', labels)
            return tf.contrib.losses.mean_pairwise_squared_error(predictions, labels, weight)
        reconstruction_similarity_loss_fn = masked_mpse
    elif method == 'mse':
        reconstruction_similarity_loss_fn = tf.contrib.losses.mean_squared_error
    elif method == 'hinged_mse':

        def hinged_mse(predictions, labels, weight):
            if False:
                while True:
                    i = 10
            diffs = tf.square(predictions - labels)
            diffs = tf.maximum(0.0, diffs - max_diff)
            return tf.reduce_mean(diffs) * weight
        reconstruction_similarity_loss_fn = hinged_mse
    elif method == 'hinged_mae':

        def hinged_mae(predictions, labels, weight):
            if False:
                i = 10
                return i + 15
            diffs = tf.abs(predictions - labels)
            diffs = tf.maximum(0.0, diffs - max_diff)
            return tf.reduce_mean(diffs) * weight
        reconstruction_similarity_loss_fn = hinged_mae
    else:
        raise ValueError('Unknown reconstruction loss %s' % method)
    reconstruction_similarity_loss = reconstruction_similarity_loss_fn(reconstructions, source_images, weight)
    name = '%s_Similarity_(%s)' % (name, method)
    tf.summary.scalar(name, reconstruction_similarity_loss)
    return reconstruction_similarity_loss

def g_step_loss(source_images, source_labels, end_points, hparams, num_classes):
    if False:
        for i in range(10):
            print('nop')
    "Configures the loss function which runs during the g-step.\n\n  Args:\n    source_images: A `Tensor` of shape [batch_size, height, width, channels].\n    source_labels: A dictionary of `Tensors` of shape [batch_size]. Valid keys\n      are 'class' and 'quaternion'.\n    end_points: A map of the network end points.\n    hparams: The hyperparameters struct.\n    num_classes: Number of classes for classifier loss\n\n  Returns:\n    A `Tensor` representing a loss function.\n\n  Raises:\n    ValueError: if hparams.transferred_similarity_loss_weight is non-zero but\n      hparams.transferred_similarity_loss is invalid.\n  "
    generator_loss = 0
    style_transfer_loss = tf.losses.sigmoid_cross_entropy(logits=end_points['transferred_domain_logits'], multi_class_labels=tf.ones_like(end_points['transferred_domain_logits']), weights=hparams.style_transfer_loss_weight)
    tf.summary.scalar('Style_transfer_loss', style_transfer_loss)
    generator_loss += style_transfer_loss
    generator_loss += _transferred_similarity_loss(end_points['transferred_images'], source_images, weight=hparams.transferred_similarity_loss_weight, method=hparams.transferred_similarity_loss, name='transferred_similarity')
    if source_labels is not None and hparams.task_tower_in_g_step:
        generator_loss += _add_task_specific_losses(end_points, source_labels, num_classes, hparams) * hparams.task_loss_in_g_weight
    return generator_loss

def d_step_loss(end_points, source_labels, num_classes, hparams):
    if False:
        while True:
            i = 10
    'Configures the losses during the D-Step.\n\n  Note that during the D-step, the model optimizes both the domain (binary)\n  classifier and the task classifier.\n\n  Args:\n    end_points: A map of the network end points.\n    source_labels: A dictionary of output labels to `Tensors`.\n    num_classes: The number of classes used by the classifier.\n    hparams: The hyperparameters struct.\n\n  Returns:\n    A `Tensor` representing the value of the D-step loss.\n  '
    domain_classifier_loss = add_domain_classifier_losses(end_points, hparams)
    task_classifier_loss = 0
    if source_labels is not None:
        task_classifier_loss = _add_task_specific_losses(end_points, source_labels, num_classes, hparams, add_summaries=True)
    return domain_classifier_loss + task_classifier_loss