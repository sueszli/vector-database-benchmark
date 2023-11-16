"""A library to train Inception using multiple replicas with synchronous update.

Please see accompanying README.md for details and instructions.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import os.path
import time
import numpy as np
import tensorflow as tf
from inception import image_processing
from inception import inception_model as inception
from inception.slim import slim
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('job_name', '', 'One of "ps", "worker"')
tf.app.flags.DEFINE_string('ps_hosts', '', "Comma-separated list of hostname:port for the parameter server jobs. e.g. 'machine1:2222,machine2:1111,machine2:2222'")
tf.app.flags.DEFINE_string('worker_hosts', '', "Comma-separated list of hostname:port for the worker jobs. e.g. 'machine1:2222,machine2:1111,machine2:2222'")
tf.app.flags.DEFINE_string('protocol', 'grpc', 'Communication protocol to use in distributed execution (default grpc) ')
tf.app.flags.DEFINE_string('train_dir', '/tmp/imagenet_train', 'Directory where to write event logs and checkpoint.')
tf.app.flags.DEFINE_integer('max_steps', 1000000, 'Number of batches to run.')
tf.app.flags.DEFINE_string('subset', 'train', 'Either "train" or "validation".')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Whether to log device placement.')
tf.app.flags.DEFINE_integer('task_id', 0, 'Task ID of the worker/replica running the training.')
tf.app.flags.DEFINE_integer('num_replicas_to_aggregate', -1, 'Number of gradients to collect before updating the parameters.')
tf.app.flags.DEFINE_integer('save_interval_secs', 10 * 60, 'Save interval seconds.')
tf.app.flags.DEFINE_integer('save_summaries_secs', 180, 'Save summaries interval seconds.')
tf.app.flags.DEFINE_float('initial_learning_rate', 0.045, 'Initial learning rate.')
tf.app.flags.DEFINE_float('num_epochs_per_decay', 2.0, 'Epochs after which learning rate decays.')
tf.app.flags.DEFINE_float('learning_rate_decay_factor', 0.94, 'Learning rate decay factor.')
RMSPROP_DECAY = 0.9
RMSPROP_MOMENTUM = 0.9
RMSPROP_EPSILON = 1.0

def train(target, dataset, cluster_spec):
    if False:
        return 10
    'Train Inception on a dataset for a number of steps.'
    num_workers = len(cluster_spec.as_dict()['worker'])
    num_parameter_servers = len(cluster_spec.as_dict()['ps'])
    if FLAGS.num_replicas_to_aggregate == -1:
        num_replicas_to_aggregate = num_workers
    else:
        num_replicas_to_aggregate = FLAGS.num_replicas_to_aggregate
    assert num_workers > 0 and num_parameter_servers > 0, ' num_workers and num_parameter_servers must be > 0.'
    is_chief = FLAGS.task_id == 0
    with tf.device('/job:worker/task:%d' % FLAGS.task_id):
        with slim.scopes.arg_scope([slim.variables.variable, slim.variables.global_step], device=slim.variables.VariableDeviceChooser(num_parameter_servers)):
            global_step = slim.variables.global_step()
            num_batches_per_epoch = dataset.num_examples_per_epoch() / FLAGS.batch_size
            decay_steps = int(num_batches_per_epoch * FLAGS.num_epochs_per_decay / num_replicas_to_aggregate)
            lr = tf.train.exponential_decay(FLAGS.initial_learning_rate, global_step, decay_steps, FLAGS.learning_rate_decay_factor, staircase=True)
            tf.summary.scalar('learning_rate', lr)
            opt = tf.train.RMSPropOptimizer(lr, RMSPROP_DECAY, momentum=RMSPROP_MOMENTUM, epsilon=RMSPROP_EPSILON)
            (images, labels) = image_processing.distorted_inputs(dataset, batch_size=FLAGS.batch_size, num_preprocess_threads=FLAGS.num_preprocess_threads)
            num_classes = dataset.num_classes() + 1
            logits = inception.inference(images, num_classes, for_training=True)
            inception.loss(logits, labels)
            losses = tf.get_collection(slim.losses.LOSSES_COLLECTION)
            losses += tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            total_loss = tf.add_n(losses, name='total_loss')
            if is_chief:
                loss_averages = tf.train.ExponentialMovingAverage(0.9, name='avg')
                loss_averages_op = loss_averages.apply(losses + [total_loss])
                for l in losses + [total_loss]:
                    loss_name = l.op.name
                    tf.summary.scalar(loss_name + ' (raw)', l)
                    tf.summary.scalar(loss_name, loss_averages.average(l))
                with tf.control_dependencies([loss_averages_op]):
                    total_loss = tf.identity(total_loss)
            exp_moving_averager = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY, global_step)
            variables_to_average = tf.trainable_variables() + tf.moving_average_variables()
            for var in variables_to_average:
                tf.summary.histogram(var.op.name, var)
            opt = tf.train.SyncReplicasOptimizer(opt, replicas_to_aggregate=num_replicas_to_aggregate, total_num_replicas=num_workers, variable_averages=exp_moving_averager, variables_to_average=variables_to_average)
            batchnorm_updates = tf.get_collection(slim.ops.UPDATE_OPS_COLLECTION)
            assert batchnorm_updates, 'Batchnorm updates are missing'
            batchnorm_updates_op = tf.group(*batchnorm_updates)
            with tf.control_dependencies([batchnorm_updates_op]):
                total_loss = tf.identity(total_loss)
            grads = opt.compute_gradients(total_loss)
            for (grad, var) in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)
            apply_gradients_op = opt.apply_gradients(grads, global_step=global_step)
            with tf.control_dependencies([apply_gradients_op]):
                train_op = tf.identity(total_loss, name='train_op')
            chief_queue_runners = [opt.get_chief_queue_runner()]
            init_tokens_op = opt.get_init_tokens_op()
            saver = tf.train.Saver()
            summary_op = tf.summary.merge_all()
            init_op = tf.global_variables_initializer()
            sv = tf.train.Supervisor(is_chief=is_chief, logdir=FLAGS.train_dir, init_op=init_op, summary_op=None, global_step=global_step, saver=saver, save_model_secs=FLAGS.save_interval_secs)
            tf.logging.info('%s Supervisor' % datetime.now())
            sess_config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=FLAGS.log_device_placement)
            sess = sv.prepare_or_wait_for_session(target, config=sess_config)
            queue_runners = tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS)
            sv.start_queue_runners(sess, queue_runners)
            tf.logging.info('Started %d queues for processing input data.', len(queue_runners))
            if is_chief:
                sv.start_queue_runners(sess, chief_queue_runners)
                sess.run(init_tokens_op)
            next_summary_time = time.time() + FLAGS.save_summaries_secs
            while not sv.should_stop():
                try:
                    start_time = time.time()
                    (loss_value, step) = sess.run([train_op, global_step])
                    assert not np.isnan(loss_value), 'Model diverged with loss = NaN'
                    if step > FLAGS.max_steps:
                        break
                    duration = time.time() - start_time
                    if step % 30 == 0:
                        examples_per_sec = FLAGS.batch_size / float(duration)
                        format_str = 'Worker %d: %s: step %d, loss = %.2f(%.1f examples/sec; %.3f  sec/batch)'
                        tf.logging.info(format_str % (FLAGS.task_id, datetime.now(), step, loss_value, examples_per_sec, duration))
                    if is_chief and next_summary_time < time.time():
                        tf.logging.info('Running Summary operation on the chief.')
                        summary_str = sess.run(summary_op)
                        sv.summary_computed(sess, summary_str)
                        tf.logging.info('Finished running Summary operation.')
                        next_summary_time += FLAGS.save_summaries_secs
                except:
                    if is_chief:
                        tf.logging.info('Chief got exception while running!')
                    raise
            sv.stop()
            if is_chief:
                saver.save(sess, os.path.join(FLAGS.train_dir, 'model.ckpt'), global_step=global_step)