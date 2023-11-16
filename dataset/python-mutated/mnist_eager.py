"""MNIST model training with TensorFlow eager execution.

See:
https://research.googleblog.com/2017/10/eager-execution-imperative-define-by.html

This program demonstrates training of the convolutional neural network model
defined in mnist.py with eager execution enabled.

If you are not interested in eager execution, you should ignore this file.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from absl import app as absl_app
from absl import flags
import tensorflow as tf
from tensorflow.python import eager as tfe
from official.r1.mnist import dataset as mnist_dataset
from official.r1.mnist import mnist
from official.utils.flags import core as flags_core
from official.utils.misc import model_helpers

def loss(logits, labels):
    if False:
        i = 10
        return i + 15
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=labels))

def compute_accuracy(logits, labels):
    if False:
        return 10
    predictions = tf.argmax(logits, axis=1, output_type=tf.int64)
    labels = tf.cast(labels, tf.int64)
    batch_size = int(logits.shape[0])
    return tf.reduce_sum(tf.cast(tf.equal(predictions, labels), dtype=tf.float32)) / batch_size

def train(model, optimizer, dataset, step_counter, log_interval=None):
    if False:
        print('Hello World!')
    'Trains model on `dataset` using `optimizer`.'
    start = time.time()
    for (batch, (images, labels)) in enumerate(dataset):
        with tf.contrib.summary.record_summaries_every_n_global_steps(10, global_step=step_counter):
            with tf.GradientTape() as tape:
                logits = model(images, training=True)
                loss_value = loss(logits, labels)
                tf.contrib.summary.scalar('loss', loss_value)
                tf.contrib.summary.scalar('accuracy', compute_accuracy(logits, labels))
            grads = tape.gradient(loss_value, model.variables)
            optimizer.apply_gradients(zip(grads, model.variables), global_step=step_counter)
            if log_interval and batch % log_interval == 0:
                rate = log_interval / (time.time() - start)
                print('Step #%d\tLoss: %.6f (%d steps/sec)' % (batch, loss_value, rate))
                start = time.time()

def test(model, dataset):
    if False:
        while True:
            i = 10
    'Perform an evaluation of `model` on the examples from `dataset`.'
    avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
    accuracy = tf.keras.metrics.Accuracy('accuracy', dtype=tf.float32)
    for (images, labels) in dataset:
        logits = model(images, training=False)
        avg_loss.update_state(loss(logits, labels))
        accuracy.update_state(tf.argmax(logits, axis=1, output_type=tf.int64), tf.cast(labels, tf.int64))
    print('Test set: Average loss: %.4f, Accuracy: %4f%%\n' % (avg_loss.result(), 100 * accuracy.result()))
    with tf.contrib.summary.always_record_summaries():
        tf.contrib.summary.scalar('loss', avg_loss.result())
        tf.contrib.summary.scalar('accuracy', accuracy.result())

def run_mnist_eager(flags_obj):
    if False:
        while True:
            i = 10
    'Run MNIST training and eval loop in eager mode.\n\n  Args:\n    flags_obj: An object containing parsed flag values.\n  '
    tf.enable_eager_execution()
    model_helpers.apply_clean(flags.FLAGS)
    (device, data_format) = ('/gpu:0', 'channels_first')
    if flags_obj.no_gpu or not tf.test.is_gpu_available():
        (device, data_format) = ('/cpu:0', 'channels_last')
    if flags_obj.data_format is not None:
        data_format = flags_obj.data_format
    print('Using device %s, and data format %s.' % (device, data_format))
    train_ds = mnist_dataset.train(flags_obj.data_dir).shuffle(60000).batch(flags_obj.batch_size)
    test_ds = mnist_dataset.test(flags_obj.data_dir).batch(flags_obj.batch_size)
    model = mnist.create_model(data_format)
    optimizer = tf.train.MomentumOptimizer(flags_obj.lr, flags_obj.momentum)
    if flags_obj.output_dir:
        train_dir = os.path.join(flags_obj.output_dir, 'train')
        test_dir = os.path.join(flags_obj.output_dir, 'eval')
        tf.gfile.MakeDirs(flags_obj.output_dir)
    else:
        train_dir = None
        test_dir = None
    summary_writer = tf.contrib.summary.create_file_writer(train_dir, flush_millis=10000)
    test_summary_writer = tf.contrib.summary.create_file_writer(test_dir, flush_millis=10000, name='test')
    checkpoint_prefix = os.path.join(flags_obj.model_dir, 'ckpt')
    step_counter = tf.train.get_or_create_global_step()
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer, step_counter=step_counter)
    checkpoint.restore(tf.train.latest_checkpoint(flags_obj.model_dir))
    with tf.device(device):
        for _ in range(flags_obj.train_epochs):
            start = time.time()
            with summary_writer.as_default():
                train(model, optimizer, train_ds, step_counter, flags_obj.log_interval)
            end = time.time()
            print('\nTrain time for epoch #%d (%d total steps): %f' % (checkpoint.save_counter.numpy() + 1, step_counter.numpy(), end - start))
            with test_summary_writer.as_default():
                test(model, test_ds)
            checkpoint.save(checkpoint_prefix)

def define_mnist_eager_flags():
    if False:
        while True:
            i = 10
    'Defined flags and defaults for MNIST in eager mode.'
    flags_core.define_base(clean=True, train_epochs=True, export_dir=True, distribution_strategy=True)
    flags_core.define_image()
    flags.adopt_module_key_flags(flags_core)
    flags.DEFINE_integer(name='log_interval', short_name='li', default=10, help=flags_core.help_wrap('batches between logging training status'))
    flags.DEFINE_string(name='output_dir', short_name='od', default=None, help=flags_core.help_wrap('Directory to write TensorBoard summaries'))
    flags.DEFINE_float(name='learning_rate', short_name='lr', default=0.01, help=flags_core.help_wrap('Learning rate.'))
    flags.DEFINE_float(name='momentum', short_name='m', default=0.5, help=flags_core.help_wrap('SGD momentum.'))
    flags.DEFINE_bool(name='no_gpu', short_name='nogpu', default=False, help=flags_core.help_wrap('disables GPU usage even if a GPU is available'))
    flags_core.set_defaults(data_dir='/tmp/tensorflow/mnist/input_data', model_dir='/tmp/tensorflow/mnist/checkpoints/', batch_size=100, train_epochs=10)

def main(_):
    if False:
        return 10
    run_mnist_eager(flags.FLAGS)
if __name__ == '__main__':
    define_mnist_eager_flags()
    absl_app.run(main=main)