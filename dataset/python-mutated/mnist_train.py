"""MNIST model float training script with TensorFlow graph execution."""
import os
from absl import flags
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.compiler.mlir.tfr.examples.mnist import gen_mnist_ops
from tensorflow.compiler.mlir.tfr.examples.mnist import ops_defs
from tensorflow.python.framework import load_library
flags.DEFINE_integer('train_steps', 200, 'Number of steps in training.')
_lib_dir = os.path.dirname(gen_mnist_ops.__file__)
_lib_name = os.path.basename(gen_mnist_ops.__file__)[4:].replace('.py', '.so')
load_library.load_op_library(os.path.join(_lib_dir, _lib_name))
num_classes = 10
num_features = 784
num_channels = 1
learning_rate = 0.001
display_step = 10
batch_size = 32
n_hidden_1 = 32
n_hidden_2 = 64
n_hidden_3 = 64
flatten_size = num_features // 16 * n_hidden_2
seed = 66478

class FloatModel(tf.Module):
    """Float inference for mnist model."""

    def __init__(self):
        if False:
            print('Hello World!')
        self.weights = {'f1': tf.Variable(tf.random.truncated_normal([5, 5, num_channels, n_hidden_1], stddev=0.1, seed=seed)), 'f2': tf.Variable(tf.random.truncated_normal([5, 5, n_hidden_1, n_hidden_2], stddev=0.1, seed=seed)), 'f3': tf.Variable(tf.random.truncated_normal([n_hidden_3, flatten_size], stddev=0.1, seed=seed)), 'f4': tf.Variable(tf.random.truncated_normal([num_classes, n_hidden_3], stddev=0.1, seed=seed))}
        self.biases = {'b1': tf.Variable(tf.zeros([n_hidden_1])), 'b2': tf.Variable(tf.zeros([n_hidden_2])), 'b3': tf.Variable(tf.zeros([n_hidden_3])), 'b4': tf.Variable(tf.zeros([num_classes]))}

    @tf.function
    def __call__(self, data):
        if False:
            while True:
                i = 10
        'The Model definition.'
        x = tf.reshape(data, [-1, 28, 28, 1])
        conv1 = gen_mnist_ops.new_conv2d(x, self.weights['f1'], self.biases['b1'], 1, 1, 1, 1, 'SAME', 'RELU')
        max_pool1 = gen_mnist_ops.new_max_pool(conv1, 2, 2, 2, 2, 'SAME')
        conv2 = gen_mnist_ops.new_conv2d(max_pool1, self.weights['f2'], self.biases['b2'], 1, 1, 1, 1, 'SAME', 'RELU')
        max_pool2 = gen_mnist_ops.new_max_pool(conv2, 2, 2, 2, 2, 'SAME')
        reshape = tf.reshape(max_pool2, [-1, flatten_size])
        fc1 = gen_mnist_ops.new_fully_connected(reshape, self.weights['f3'], self.biases['b3'], 'RELU')
        return gen_mnist_ops.new_fully_connected(fc1, self.weights['f4'], self.biases['b4'])

def main(strategy):
    if False:
        i = 10
        return i + 15
    'Trains an MNIST model using the given tf.distribute.Strategy.'
    os.environ['TF_MLIR_TFR_LIB_DIR'] = 'tensorflow/compiler/mlir/tfr/examples/mnist'
    ds_train = tfds.load('mnist', split='train', shuffle_files=True)
    ds_train = ds_train.shuffle(1024).batch(batch_size).prefetch(64)
    ds_train = strategy.experimental_distribute_dataset(ds_train)
    with strategy.scope():
        model = FloatModel()
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    def train_step(features):
        if False:
            i = 10
            return i + 15
        inputs = tf.image.convert_image_dtype(features['image'], dtype=tf.float32, saturate=False)
        labels = tf.one_hot(features['label'], num_classes)
        with tf.GradientTape() as tape:
            logits = model(inputs)
            loss_value = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels, logits))
        grads = tape.gradient(loss_value, model.trainable_variables)
        correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return (accuracy, loss_value)

    @tf.function
    def distributed_train_step(dist_inputs):
        if False:
            for i in range(10):
                print('nop')
        (per_replica_accuracy, per_replica_losses) = strategy.run(train_step, args=(dist_inputs,))
        accuracy = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_accuracy, axis=None)
        loss_value = strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)
        return (accuracy, loss_value)
    iterator = iter(ds_train)
    accuracy = 0.0
    for step in range(flags.FLAGS.train_steps):
        (accuracy, loss_value) = distributed_train_step(next(iterator))
        if step % display_step == 0:
            tf.print('Step %d:' % step)
            tf.print('    Loss = %f' % loss_value)
            tf.print('    Batch accuracy = %f' % accuracy)
    return accuracy