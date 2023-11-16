"""Evaluation for CIFAR-10.

Accuracy:
cifar10_train.py achieves 83.0% accuracy after 100K steps (256 epochs
of data) as judged by cifar10_eval.py.

Speed:
On a single Tesla K40, cifar10_train.py processes a single batch of 128 images
in 0.25-0.35 sec (i.e. 350 - 600 images /sec). The model reaches ~86%
accuracy after 100K steps in 8 hours of training time.

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import time
import numpy as np
import tensorflow as tf
import cifar10
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/tmp/cifar10_eval', 'Directory where to write event logs.')
tf.app.flags.DEFINE_string('eval_data', 'test', "Either 'test' or 'train_eval'.")
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/cifar10_train', 'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 5, 'How often to run the eval.')
tf.app.flags.DEFINE_integer('num_examples', 1000, 'Number of examples to run.')
tf.app.flags.DEFINE_boolean('run_once', False, 'Whether to run eval only once.')

def eval_once(saver, summary_writer, top_k_op, summary_op):
    if False:
        print('Hello World!')
    'Run Eval once.\n\n  Args:\n    saver: Saver.\n    summary_writer: Summary writer.\n    top_k_op: Top K op.\n    summary_op: Summary op.\n  '
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
        else:
            print('No checkpoint file found')
            return
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(float(FLAGS.num_examples) / FLAGS.batch_size))
            true_count = 0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            while step < num_iter and (not coord.should_stop()):
                predictions = sess.run([top_k_op])
                true_count += np.sum(predictions)
                step += 1
            precision = true_count / total_sample_count
            print('%s: precision @ 1 = %.3f' % (datetime.now(), precision))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate():
    if False:
        return 10
    'Eval CIFAR-10 for a number of steps.'
    with tf.Graph().as_default() as g:
        (images, labels) = cifar10.inputs(eval_data=FLAGS.eval_data)
        logits = cifar10.inference(images)
        logits = tf.cast(logits, 'float32')
        labels = tf.cast(labels, 'int32')
        top_k_op = tf.nn.in_top_k(logits, labels, 1)
        variable_averages = tf.train.ExponentialMovingAverage(cifar10.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, g)
        while True:
            eval_once(saver, summary_writer, top_k_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)

def main(argv=None):
    if False:
        print('Hello World!')
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    evaluate()
if __name__ == '__main__':
    tf.app.run()