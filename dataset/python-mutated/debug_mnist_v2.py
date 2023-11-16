"""Demo of the tfdbg curses CLI: Locating the source of bad numerical values with TF v2.

This demo contains a classical example of a neural network for the mnist
dataset, but modifications are made so that problematic numerical values (infs
and nans) appear in nodes of the graph during training.
"""
import argparse
import sys
from absl import app
import tensorflow.compat.v2 as tf
IMAGE_SIZE = 28
HIDDEN_SIZE = 500
NUM_LABELS = 10
RAND_SEED = 42
tf.compat.v1.enable_v2_behavior()
FLAGS = None

def parse_args():
    if False:
        print('Hello World!')
    'Parses commandline arguments.\n\n  Returns:\n    A tuple (parsed, unparsed) of the parsed object and a group of unparsed\n      arguments that did not match the parser.\n  '
    parser = argparse.ArgumentParser()
    parser.register('type', 'bool', lambda v: v.lower() == 'true')
    parser.add_argument('--max_steps', type=int, default=10, help='Number of steps to run trainer.')
    parser.add_argument('--train_batch_size', type=int, default=100, help='Batch size used during training.')
    parser.add_argument('--learning_rate', type=float, default=0.025, help='Initial learning rate.')
    parser.add_argument('--data_dir', type=str, default='/tmp/mnist_data', help='Directory for storing data')
    parser.add_argument('--fake_data', type='bool', nargs='?', const=True, default=False, help='Use fake MNIST data for unit testing')
    parser.add_argument('--check_numerics', type='bool', nargs='?', const=True, default=False, help='Use tfdbg to track down bad values during training. Mutually exclusive with the --dump_dir flag.')
    parser.add_argument('--dump_dir', type=str, default=None, help='Dump TensorFlow program debug data to the specified directory. The dumped data contains information regarding tf.function building, execution of ops and tf.functions, as well as their stack traces and associated source-code snapshots. Mutually exclusive with the --check_numerics flag.')
    parser.add_argument('--dump_tensor_debug_mode', type=str, default='FULL_HEALTH', help='Mode for dumping tensor values. Options: NO_TENSOR, CURT_HEALTH, CONCISE_HEALTH, SHAPE, FULL_HEALTH. This is relevant only when --dump_dir is set.')
    parser.add_argument('--dump_circular_buffer_size', type=int, default=-1, help='Size of the circular buffer used to dump execution events. A value <= 0 disables the circular-buffer behavior and causes all instrumented tensor values to be dumped. This is relevant only when --dump_dir is set.')
    parser.add_argument('--use_random_config_path', type='bool', nargs='?', const=True, default=False, help='If set, set config file path to a random file in the temporary\n      directory.')
    return parser.parse_known_args()

def main(_):
    if False:
        return 10
    if FLAGS.check_numerics and FLAGS.dump_dir:
        raise ValueError('The --check_numerics and --dump_dir flags are mutually exclusive.')
    if FLAGS.check_numerics:
        tf.debugging.enable_check_numerics()
    elif FLAGS.dump_dir:
        tf.debugging.experimental.enable_dump_debug_info(FLAGS.dump_dir, tensor_debug_mode=FLAGS.dump_tensor_debug_mode, circular_buffer_size=FLAGS.dump_circular_buffer_size)
    if FLAGS.fake_data:
        imgs = tf.random.uniform(maxval=256, shape=(1000, 28, 28), dtype=tf.int32)
        labels = tf.random.uniform(maxval=10, shape=(1000,), dtype=tf.int32)
        mnist_train = (imgs, labels)
        mnist_test = (imgs, labels)
    else:
        (mnist_train, mnist_test) = tf.keras.datasets.mnist.load_data()

    @tf.function
    def format_example(imgs, labels):
        if False:
            i = 10
            return i + 15
        'Formats each training and test example to work with our model.'
        imgs = tf.reshape(imgs, [-1, 28 * 28])
        imgs = tf.cast(imgs, tf.float32) / 255.0
        labels = tf.one_hot(labels, depth=10, dtype=tf.float32)
        return (imgs, labels)
    train_ds = tf.data.Dataset.from_tensor_slices(mnist_train).shuffle(FLAGS.train_batch_size * FLAGS.max_steps, seed=RAND_SEED).batch(FLAGS.train_batch_size)
    train_ds = train_ds.map(format_example)
    test_ds = tf.data.Dataset.from_tensor_slices(mnist_test).repeat().batch(len(mnist_test[0]))
    test_ds = test_ds.map(format_example)

    def get_dense_weights(input_dim, output_dim):
        if False:
            for i in range(10):
                print('nop')
        'Initializes the parameters for a single dense layer.'
        initial_kernel = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.1, seed=RAND_SEED)
        kernel = tf.Variable(initial_kernel([input_dim, output_dim]))
        bias = tf.Variable(tf.constant(0.1, shape=[output_dim]))
        return (kernel, bias)

    @tf.function
    def dense_layer(weights, input_tensor, act=tf.nn.relu):
        if False:
            i = 10
            return i + 15
        'Runs the forward computation for a single dense layer.'
        (kernel, bias) = weights
        preactivate = tf.matmul(input_tensor, kernel) + bias
        activations = act(preactivate)
        return activations
    hidden_weights = get_dense_weights(IMAGE_SIZE ** 2, HIDDEN_SIZE)
    output_weights = get_dense_weights(HIDDEN_SIZE, NUM_LABELS)
    variables = hidden_weights + output_weights

    @tf.function
    def model(x):
        if False:
            return 10
        'Feed forward function of the model.\n\n    Args:\n      x: a (?, 28*28) tensor consisting of the feature inputs for a batch of\n        examples.\n\n    Returns:\n      A (?, 10) tensor containing the class scores for each example.\n    '
        hidden_act = dense_layer(hidden_weights, x)
        logits_act = dense_layer(output_weights, hidden_act, tf.identity)
        y = tf.nn.softmax(logits_act)
        return y

    @tf.function
    def loss(probs, labels):
        if False:
            while True:
                i = 10
        'Calculates cross entropy loss.\n\n    Args:\n      probs: Class probabilities predicted by the model. The shape is expected\n        to be (?, 10).\n      labels: Truth labels for the classes, as one-hot encoded vectors. The\n        shape is expected to be the same as `probs`.\n\n    Returns:\n      A scalar loss tensor.\n    '
        diff = -labels * tf.math.log(probs)
        loss = tf.reduce_mean(diff)
        return loss
    train_batches = iter(train_ds)
    test_batches = iter(test_ds)
    optimizer = tf.optimizers.Adam(learning_rate=FLAGS.learning_rate)
    for i in range(FLAGS.max_steps):
        (x_train, y_train) = next(train_batches)
        (x_test, y_test) = next(test_batches)
        with tf.GradientTape() as tape:
            y = model(x_train)
            loss_val = loss(y, y_train)
        grads = tape.gradient(loss_val, variables)
        optimizer.apply_gradients(zip(grads, variables))
        y = model(x_test)
        correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_test, 1))
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print('Accuracy at step %d: %s' % (i, accuracy.numpy()))
if __name__ == '__main__':
    (FLAGS, unparsed) = parse_args()
    app.run(main=main, argv=[sys.argv[0]] + unparsed)