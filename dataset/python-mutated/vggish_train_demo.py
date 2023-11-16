"""A simple demonstration of running VGGish in training mode.

This is intended as a toy example that demonstrates how to use the VGGish model
definition within a larger model that adds more layers on top, and then train
the larger model. If you let VGGish train as well, then this allows you to
fine-tune the VGGish model parameters for your application. If you don't let
VGGish train, then you use VGGish as a feature extractor for the layers above
it.

For this toy task, we are training a classifier to distinguish between three
classes: sine waves, constant signals, and white noise. We generate synthetic
waveforms from each of these classes, convert into shuffled batches of log mel
spectrogram examples with associated labels, and feed the batches into a model
that includes VGGish at the bottom and a couple of additional layers on top. We
also plumb in labels that are associated with the examples, which feed a label
loss used for training.

Usage:
  # Run training for 100 steps using a model checkpoint in the default
  # location (vggish_model.ckpt in the current directory). Allow VGGish
  # to get fine-tuned.
  $ python vggish_train_demo.py --num_batches 100

  # Same as before but run for fewer steps and don't change VGGish parameters
  # and use a checkpoint in a different location
  $ python vggish_train_demo.py --num_batches 50 \\
                                --train_vggish=False \\
                                --checkpoint /path/to/model/checkpoint
"""
from __future__ import print_function
from random import shuffle
import numpy as np
import tensorflow as tf
import vggish_input
import vggish_params
import vggish_slim
flags = tf.app.flags
slim = tf.contrib.slim
flags.DEFINE_integer('num_batches', 30, 'Number of batches of examples to feed into the model. Each batch is of variable size and contains shuffled examples of each class of audio.')
flags.DEFINE_boolean('train_vggish', True, 'If True, allow VGGish parameters to change during training, thus fine-tuning VGGish. If False, VGGish parameters are fixed, thus using VGGish as a fixed feature extractor.')
flags.DEFINE_string('checkpoint', 'vggish_model.ckpt', 'Path to the VGGish checkpoint file.')
FLAGS = flags.FLAGS
_NUM_CLASSES = 3

def _get_examples_batch():
    if False:
        for i in range(10):
            print('nop')
    'Returns a shuffled batch of examples of all audio classes.\n\n  Note that this is just a toy function because this is a simple demo intended\n  to illustrate how the training code might work.\n\n  Returns:\n    a tuple (features, labels) where features is a NumPy array of shape\n    [batch_size, num_frames, num_bands] where the batch_size is variable and\n    each row is a log mel spectrogram patch of shape [num_frames, num_bands]\n    suitable for feeding VGGish, while labels is a NumPy array of shape\n    [batch_size, num_classes] where each row is a multi-hot label vector that\n    provides the labels for corresponding rows in features.\n  '
    num_seconds = 5
    sr = 44100
    t = np.linspace(0, num_seconds, int(num_seconds * sr))
    freq = np.random.uniform(100, 1000)
    sine = np.sin(2 * np.pi * freq * t)
    magnitude = np.random.uniform(-1, 1)
    const = magnitude * t
    noise = np.random.normal(-1, 1, size=t.shape)
    sine_examples = vggish_input.waveform_to_examples(sine, sr)
    sine_labels = np.array([[1, 0, 0]] * sine_examples.shape[0])
    const_examples = vggish_input.waveform_to_examples(const, sr)
    const_labels = np.array([[0, 1, 0]] * const_examples.shape[0])
    noise_examples = vggish_input.waveform_to_examples(noise, sr)
    noise_labels = np.array([[0, 0, 1]] * noise_examples.shape[0])
    all_examples = np.concatenate((sine_examples, const_examples, noise_examples))
    all_labels = np.concatenate((sine_labels, const_labels, noise_labels))
    labeled_examples = list(zip(all_examples, all_labels))
    shuffle(labeled_examples)
    features = [example for (example, _) in labeled_examples]
    labels = [label for (_, label) in labeled_examples]
    return (features, labels)

def main(_):
    if False:
        for i in range(10):
            print('nop')
    with tf.Graph().as_default(), tf.Session() as sess:
        embeddings = vggish_slim.define_vggish_slim(FLAGS.train_vggish)
        with tf.variable_scope('mymodel'):
            num_units = 100
            fc = slim.fully_connected(embeddings, num_units)
            logits = slim.fully_connected(fc, _NUM_CLASSES, activation_fn=None, scope='logits')
            tf.sigmoid(logits, name='prediction')
            with tf.variable_scope('train'):
                global_step = tf.Variable(0, name='global_step', trainable=False, collections=[tf.GraphKeys.GLOBAL_VARIABLES, tf.GraphKeys.GLOBAL_STEP])
                labels = tf.placeholder(tf.float32, shape=(None, _NUM_CLASSES), name='labels')
                xent = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels, name='xent')
                loss = tf.reduce_mean(xent, name='loss_op')
                tf.summary.scalar('loss', loss)
                optimizer = tf.train.AdamOptimizer(learning_rate=vggish_params.LEARNING_RATE, epsilon=vggish_params.ADAM_EPSILON)
                optimizer.minimize(loss, global_step=global_step, name='train_op')
        sess.run(tf.global_variables_initializer())
        vggish_slim.load_vggish_slim_checkpoint(sess, FLAGS.checkpoint)
        features_tensor = sess.graph.get_tensor_by_name(vggish_params.INPUT_TENSOR_NAME)
        labels_tensor = sess.graph.get_tensor_by_name('mymodel/train/labels:0')
        global_step_tensor = sess.graph.get_tensor_by_name('mymodel/train/global_step:0')
        loss_tensor = sess.graph.get_tensor_by_name('mymodel/train/loss_op:0')
        train_op = sess.graph.get_operation_by_name('mymodel/train/train_op')
        for _ in range(FLAGS.num_batches):
            (features, labels) = _get_examples_batch()
            [num_steps, loss, _] = sess.run([global_step_tensor, loss_tensor, train_op], feed_dict={features_tensor: features, labels_tensor: labels})
            print('Step %d: loss %g' % (num_steps, loss))
if __name__ == '__main__':
    tf.app.run()