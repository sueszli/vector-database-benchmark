"""A library to evaluate Inception on a single GPU.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from datetime import datetime
import math
import os.path
import time
import numpy as np
import tensorflow as tf
from inception import image_processing
from inception import inception_model as inception
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('eval_dir', '/tmp/imagenet_eval', 'Directory where to write event logs.')
tf.app.flags.DEFINE_string('checkpoint_dir', '/tmp/imagenet_train', 'Directory where to read model checkpoints.')
tf.app.flags.DEFINE_integer('eval_interval_secs', 60 * 5, 'How often to run the eval.')
tf.app.flags.DEFINE_boolean('run_once', False, 'Whether to run eval only once.')
tf.app.flags.DEFINE_integer('num_examples', 50000, 'Number of examples to run. Note that the eval ImageNet dataset contains 50000 examples.')
tf.app.flags.DEFINE_string('subset', 'validation', "Either 'validation' or 'train'.")

def _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op):
    if False:
        for i in range(10):
            print('nop')
    'Runs Eval once.\n\n  Args:\n    saver: Saver.\n    summary_writer: Summary writer.\n    top_1_op: Top 1 op.\n    top_5_op: Top 5 op.\n    summary_op: Summary op.\n  '
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(FLAGS.checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            if os.path.isabs(ckpt.model_checkpoint_path):
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                saver.restore(sess, os.path.join(FLAGS.checkpoint_dir, ckpt.model_checkpoint_path))
            global_step = ckpt.model_checkpoint_path.split('/')[-1].split('-')[-1]
            print('Successfully loaded model from %s at step=%s.' % (ckpt.model_checkpoint_path, global_step))
        else:
            print('No checkpoint file found')
            return
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord=coord, daemon=True, start=True))
            num_iter = int(math.ceil(FLAGS.num_examples / FLAGS.batch_size))
            count_top_1 = 0.0
            count_top_5 = 0.0
            total_sample_count = num_iter * FLAGS.batch_size
            step = 0
            print('%s: starting evaluation on (%s).' % (datetime.now(), FLAGS.subset))
            start_time = time.time()
            while step < num_iter and (not coord.should_stop()):
                (top_1, top_5) = sess.run([top_1_op, top_5_op])
                count_top_1 += np.sum(top_1)
                count_top_5 += np.sum(top_5)
                step += 1
                if step % 20 == 0:
                    duration = time.time() - start_time
                    sec_per_batch = duration / 20.0
                    examples_per_sec = FLAGS.batch_size / sec_per_batch
                    print('%s: [%d batches out of %d] (%.1f examples/sec; %.3fsec/batch)' % (datetime.now(), step, num_iter, examples_per_sec, sec_per_batch))
                    start_time = time.time()
            precision_at_1 = count_top_1 / total_sample_count
            recall_at_5 = count_top_5 / total_sample_count
            print('%s: precision @ 1 = %.4f recall @ 5 = %.4f [%d examples]' % (datetime.now(), precision_at_1, recall_at_5, total_sample_count))
            summary = tf.Summary()
            summary.ParseFromString(sess.run(summary_op))
            summary.value.add(tag='Precision @ 1', simple_value=precision_at_1)
            summary.value.add(tag='Recall @ 5', simple_value=recall_at_5)
            summary_writer.add_summary(summary, global_step)
        except Exception as e:
            coord.request_stop(e)
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs=10)

def evaluate(dataset):
    if False:
        print('Hello World!')
    'Evaluate model on Dataset for a number of steps.'
    with tf.Graph().as_default():
        (images, labels) = image_processing.inputs(dataset)
        num_classes = dataset.num_classes() + 1
        (logits, _) = inception.inference(images, num_classes)
        top_1_op = tf.nn.in_top_k(logits, labels, 1)
        top_5_op = tf.nn.in_top_k(logits, labels, 5)
        variable_averages = tf.train.ExponentialMovingAverage(inception.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)
        summary_op = tf.summary.merge_all()
        graph_def = tf.get_default_graph().as_graph_def()
        summary_writer = tf.summary.FileWriter(FLAGS.eval_dir, graph_def=graph_def)
        while True:
            _eval_once(saver, summary_writer, top_1_op, top_5_op, summary_op)
            if FLAGS.run_once:
                break
            time.sleep(FLAGS.eval_interval_secs)