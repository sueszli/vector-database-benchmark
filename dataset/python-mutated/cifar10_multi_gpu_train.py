"""A binary to train CIFAR-10 using multiple GPUs with synchronous updates.

Accuracy:
cifar10_multi_gpu_train.py achieves ~86% accuracy after 100K steps (256
epochs of data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
--------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)
2 Tesla K20m  | 0.13-0.20              | ~84% at 30K steps  (2.5 hours)
3 Tesla K20m  | 0.13-0.18              | ~84% at 30K steps
4 Tesla K20m  | ~0.10                  | ~84% at 30K steps

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os.path
import re
import time
from datetime import datetime
import numpy as np
import tensorflow as tf
from six.moves import xrange
import cifar10
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train', 'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_integer('num_gpus', 1, 'How many GPUs to use.')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')

def tower_loss(scope, images, labels):
    if False:
        i = 10
        return i + 15
    "Calculate the total loss on a single tower running the CIFAR model.\n\n  Args:\n    scope: unique prefix string identifying the CIFAR tower, e.g. 'tower_0'\n    images: Images. 4D tensor of shape [batch_size, height, width, 3].\n    labels: Labels. 1D tensor of shape [batch_size].\n\n  Returns:\n     Tensor of shape [] containing the total loss for a batch of data\n  "
    logits = cifar10.inference(images)
    _ = cifar10.loss(logits, labels)
    losses = tf.get_collection('losses', scope)
    total_loss = tf.add_n(losses, name='total_loss')
    for l in losses + [total_loss]:
        loss_name = re.sub('%s_[0-9]*/' % cifar10.TOWER_NAME, '', l.op.name)
        tf.summary.scalar(loss_name, l)
    return total_loss

def average_gradients(tower_grads):
    if False:
        i = 10
        return i + 15
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

def train():
    if False:
        for i in range(10):
            print('nop')
    'Train CIFAR-10 for a number of steps.'
    with tf.Graph().as_default(), tf.device('/cpu:0'):
        global_step = tf.get_variable('global_step', [], initializer=tf.constant_initializer(0), trainable=False)
        num_batches_per_epoch = cifar10.NUM_EXAMPLES_PER_EPOCH_FOR_TRAIN / FLAGS.batch_size / FLAGS.num_gpus
        decay_steps = int(num_batches_per_epoch * cifar10.NUM_EPOCHS_PER_DECAY)
        lr = tf.train.exponential_decay(cifar10.INITIAL_LEARNING_RATE, global_step, decay_steps, cifar10.LEARNING_RATE_DECAY_FACTOR, staircase=True)
        opt = tf.train.GradientDescentOptimizer(lr)
        (images, labels) = cifar10.distorted_inputs()
        images = tf.reshape(images, [cifar10.FLAGS.batch_size, 24, 24, 3])
        labels = tf.reshape(labels, [cifar10.FLAGS.batch_size])
        batch_queue = tf.contrib.slim.prefetch_queue.prefetch_queue([images, labels], capacity=2 * FLAGS.num_gpus)
        tower_grads = []
        with tf.variable_scope(tf.get_variable_scope()):
            for i in xrange(FLAGS.num_gpus):
                with tf.device('/gpu:%d' % i):
                    with tf.name_scope('%s_%d' % (cifar10.TOWER_NAME, i)) as scope:
                        (image_batch, label_batch) = batch_queue.dequeue()
                        loss = tower_loss(scope, image_batch, label_batch)
                        tf.get_variable_scope().reuse_variables()
                        summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope)
                        grads = opt.compute_gradients(loss)
                        tower_grads.append(grads)
        grads = average_gradients(tower_grads)
        summaries.append(tf.summary.scalar('learning_rate', lr))
        for (grad, var) in grads:
            if grad is not None:
                summaries.append(tf.summary.histogram(var.op.name + '/gradients', grad))
        apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)
        for var in tf.trainable_variables():
            summaries.append(tf.summary.histogram(var.op.name, var))
        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY, global_step)
        variables_averages_op = variable_averages.apply(tf.trainable_variables())
        train_op = tf.group(apply_gradient_op, variables_averages_op)
        saver = tf.train.Saver(tf.global_variables())
        summary_op = tf.summary.merge(summaries)
        init = tf.global_variables_initializer()
        sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement))
        sess.run(init)
        tf.train.start_queue_runners(sess=sess)
        summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
        for step in xrange(FLAGS.max_steps):
            start_time = time.time()
            (_, loss_value) = sess.run([train_op, loss])
            duration = time.time() - start_time
            assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
            if step % 10 == 0:
                num_examples_per_step = FLAGS.batch_size * FLAGS.num_gpus
                examples_per_sec = num_examples_per_step / duration
                sec_per_batch = duration / FLAGS.num_gpus
                format_str = '%s: step %d, loss = %.2f (%.1f examples/sec; %.3f sec/batch)'
                print(format_str % (datetime.now(), step, loss_value, examples_per_sec, sec_per_batch))
            if step % 100 == 0:
                summary_str = sess.run(summary_op)
                summary_writer.add_summary(summary_str, step)
            if step % 1000 == 0 or step + 1 == FLAGS.max_steps:
                checkpoint_path = os.path.join(FLAGS.train_dir, 'model.ckpt')
                saver.save(sess, checkpoint_path, global_step=step)

def main(argv=None):
    if False:
        i = 10
        return i + 15
    if tf.gfile.Exists(FLAGS.train_dir):
        tf.gfile.DeleteRecursively(FLAGS.train_dir)
    tf.gfile.MakeDirs(FLAGS.train_dir)
    train()
if __name__ == '__main__':
    tf.app.run()