"""A library to train Inception using multiple GPUs with synchronous updates.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
from datetime import datetime
import os.path
import re
import time
import numpy as np
import tensorflow as tf
from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train', 'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_integer('max_steps', 10000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', "Either 'train' or 'validation'.")
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_boolean('fine_tune', False, 'If set, randomly initialize the final layer of weights in order to train the network on a new task.')
tf.app.flags.DEFINE_string('pretrained_model_checkpoint_path', '', 'If specified, restore this pretrained model before beginning any training.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.1, 'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 30.0, 'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.16, 'Learning rate decay factor.')
RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

def _tower_loss(images, labels, num_classes, scope, reuse_variables=None):
    if False:
        for i in range(10):
            print('nop')
    "Calculate the total loss on a single tower running the ImageNet model.\n\n  We perform 'batch splitting'. This means that we cut up a batch across\n  multiple GPUs. For instance, if the batch size = 32 and num_gpus = 2,\n  then each tower will operate on an batch of 16 images.\n\n  Args:\n    images: Images. 4D tensor of size [batch_size, FLAGS.image_size,\n                                       FLAGS.image_size, 3].\n    labels: 1-D integer Tensor of [batch_size].\n    num_classes: number of classes\n    scope: unique prefix string identifying the ImageNet tower, e.g.\n      'tower_0'.\n\n  Returns:\n     Tensor of shape [] containing the total loss for a batch of data\n  "
    restore_logits = not FLAGS.fine_tune
    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse_variables):
        logits = inception.inference(images, num_classes, for_training=True, restore_logits=restore_logits, scope=scope)
    split_batch_size = images.get_shape().as_list()[0]
    inception.loss(logits, labels, batch_size=split_batch_size)
    losses = tf.get_collection(slim.losses.LOSSES_COLLECTION, scope)
    regularization_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
    total_loss = tf.add_n(losses + regularization_losses, name='total_loss')
    loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
    loss_averages_op = loss_averages.apply(losses + [total_loss])
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % inception.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name + ' (raw)', l)
        tf.summary.scalar(loss_name, loss_averages.average(l))
    with tf.control_dependencies([loss_averages_op]):
        total_loss = tf.identity(total_loss)
    return total_loss

def _average_gradients(tower_grads):
    if False:
        return 10
    'Calculate the average gradient for each shared variable across all towers.\n\n  Note that this function provides a synchronization point across all towers.\n\n  Args:\n    tower_grads: List of lists of (gradient, variable) tuples. The outer list\n      is over individual gradients. The inner list is over the gradient\n      calculation for each tower.\n  Returns:\n     List of pairs of (gradient, variable) where the gradient has been averaged\n     across all towers.\n  '
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for (g, _) in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads

def train(dataset):
    if False:
        i = 10
        return i + 15
    'Train on dataset for a number of steps.'
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = dataset.num_examples_per_epoch() / FLAGS.batch_size
        decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay)
        lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps, FLAGS.learning_rate_decay_factor, staircase=True)
        opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY, momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)
        assert FLAGS.batch_size % FLAGS.num_gpus == 0, 'Batch size must be divisible by number of GPUs'
        split_batch_size = int(FLAGS.batch_size / FLAGS.num_gpus)
        num_preprocess_threads = FLAGS.num_preprocess_threads * FLAGS.num_gpus
        (images, labels) = image_processing.distorted_inputs(dataset, num_preprocess_threads=num_preprocess_threads)
        input_summaries = copy.copy(tf.get_collection(tf.GraphKeys.SUMMARIES))
        num_classes = dataset.num_classes() + 1
        images_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=images)
        labels_splits = tf.split(axis=0, num_or_size_splits=FLAGS.num_gpus, value=labels)
        tower_grads = []
        reuse_variables = None
        for i in range(FLAGS.num_gpus):
            with tf.device('/gpu:%d' % i):
                with tf.name_scope('%s_%d' % (inception.TOWER_NAME, i)) as scope:
                    with slim.arg_scope([slim.variables.variable], device='/cpu:0'):
                        loss = _tower_loss(images_splits[i], labels_splits[i], num_classes, scope, reuse_variables)
                    reuse_variables = True
                    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                    batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION, scope)
                    grads = opt.compute_gradients(loss)
                    tower_grads.append(grads)
        grads = _average_gradients(tower_grads)
        summaries.extend(input_summaries)
        summaries.append(tf.summary.scalar('learning_rate', lr))
        for (grad, var) in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY, global_step)
        variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
        variables_averages_op = variable_averages.apply(variables_to_average)
        batchnorm_updates_op = tf.group(*batchnorm_updates)
        train_op = tf.group(apply_gradient_op, variables_averages_op, batchnorm_updates_op)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge(summaries)
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        if FLAGS.pretrained_model_checkpoint_path:
            assert tf.gfile.Exists(FLAGS.pretrained_model_checkpoint_path)
            variables_to_restore = tf.get_collection(slim.variables.VARIABLES_TO_RESTORE)
            restorer = tf.train.Saver(variables_to_restore)
            restorer.restore(sess, FLAGS.pretrained_model_checkpoint_path)
            print('%s: Pre-trained model restored from %s' % (datetime.now(), FLAGS.pretrained_model_checkpoint_path))
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, graph=sess.graph)
        for step in range(FLAGS.max_steps):
            start_time = time.time()
            (_, loss_value) = sess.run([train_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                examples_per_sec = FLAGS.batch_size / float(duration)
                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, duration))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 5000 == 0 or step + 1 == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)