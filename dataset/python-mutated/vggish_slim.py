"""Defines the 'VGGish' model used to generate AudioSet embedding features.

The public AudioSet release (https://research.google.com/audioset/download.html)
includes 128-D features extracted from the embedding layer of a VGG-like model
that was trained on a large Google-internal YouTube dataset. Here we provide
a TF-Slim definition of the same model, without any dependences on libraries
internal to Google. We call it 'VGGish'.

Note that we only define the model up to the embedding layer, which is the
penultimate layer before the final classifier layer. We also provide various
hyperparameter values (in vggish_params.py) that were used to train this model
internally.

For comparison, here is TF-Slim's VGG definition:
https://github.com/tensorflow/models/blob/master/research/slim/nets/vgg.py
"""
import tensorflow as tf
import vggish_params as params
slim = tf.contrib.slim

def define_vggish_slim(training=False):
    if False:
        return 10
    "Defines the VGGish TensorFlow model.\n\n  All ops are created in the current default graph, under the scope 'vggish/'.\n\n  The input is a placeholder named 'vggish/input_features' of type float32 and\n  shape [batch_size, num_frames, num_bands] where batch_size is variable and\n  num_frames and num_bands are constants, and [num_frames, num_bands] represents\n  a log-mel-scale spectrogram patch covering num_bands frequency bands and\n  num_frames time frames (where each frame step is usually 10ms). This is\n  produced by computing the stabilized log(mel-spectrogram + params.LOG_OFFSET).\n  The output is an op named 'vggish/embedding' which produces the activations of\n  a 128-D embedding layer, which is usually the penultimate layer when used as\n  part of a full model with a final classifier layer.\n\n  Args:\n    training: If true, all parameters are marked trainable.\n\n  Returns:\n    The op 'vggish/embeddings'.\n  "
    with slim.arg_scope([slim.conv2d, slim.fully_connected], weights_initializer=tf.truncated_normal_initializer(stddev=params.INIT_STDDEV), biases_initializer=tf.zeros_initializer(), activation_fn=tf.nn.relu, trainable=training), slim.arg_scope([slim.conv2d], kernel_size=[3, 3], stride=1, padding='SAME'), slim.arg_scope([slim.max_pool2d], kernel_size=[2, 2], stride=2, padding='SAME'), tf.variable_scope('vggish'):
        features = tf.placeholder(tf.float32, shape=(None, params.NUM_FRAMES, params.NUM_BANDS), name='input_features')
        net = tf.reshape(features, [-1, params.NUM_FRAMES, params.NUM_BANDS, 1])
        net = slim.conv2d(net, 64, scope='conv1')
        net = slim.max_pool2d(net, scope='pool1')
        net = slim.conv2d(net, 128, scope='conv2')
        net = slim.max_pool2d(net, scope='pool2')
        net = slim.repeat(net, 2, slim.conv2d, 256, scope='conv3')
        net = slim.max_pool2d(net, scope='pool3')
        net = slim.repeat(net, 2, slim.conv2d, 512, scope='conv4')
        net = slim.max_pool2d(net, scope='pool4')
        net = slim.flatten(net)
        net = slim.repeat(net, 2, slim.fully_connected, 4096, scope='fc1')
        net = slim.fully_connected(net, params.EMBEDDING_SIZE, scope='fc2')
        return tf.identity(net, name='embedding')

def load_vggish_slim_checkpoint(session, checkpoint_path):
    if False:
        for i in range(10):
            print('nop')
    'Loads a pre-trained VGGish-compatible checkpoint.\n\n  This function can be used as an initialization function (referred to as\n  init_fn in TensorFlow documentation) which is called in a Session after\n  initializating all variables. When used as an init_fn, this will load\n  a pre-trained checkpoint that is compatible with the VGGish model\n  definition. Only variables defined by VGGish will be loaded.\n\n  Args:\n    session: an active TensorFlow session.\n    checkpoint_path: path to a file containing a checkpoint that is\n      compatible with the VGGish model definition.\n  '
    with tf.Graph().as_default():
        define_vggish_slim(training=False)
        vggish_var_names = [v.name for v in tf.global_variables()]
    vggish_vars = [v for v in tf.global_variables() if v.name in vggish_var_names]
    saver = tf.train.Saver(vggish_vars, name='vggish_load_pretrained', write_version=1)
    saver.restore(session, checkpoint_path)