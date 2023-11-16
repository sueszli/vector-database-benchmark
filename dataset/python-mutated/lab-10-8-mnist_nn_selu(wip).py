import tensorflow as tf
import random
'\nTensorflow Implementation of the Scaled ELU function and Dropout\n'
import numbers
from tensorflow.contrib import layers
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.layers import utils
from tensorflow.examples.tutorials.mnist import input_data
tf.set_random_seed(777)

def selu(x):
    if False:
        i = 10
        return i + 15
    with ops.name_scope('elu') as scope:
        alpha = 1.6732632423543772
        scale = 1.0507009873554805
        return scale * tf.where(x >= 0.0, x, alpha * tf.nn.elu(x))

def dropout_selu(x, keep_prob, alpha=-1.7580993408473766, fixedPointMean=0.0, fixedPointVar=1.0, noise_shape=None, seed=None, name=None, training=False):
    if False:
        while True:
            i = 10
    'Dropout to a value with rescaling.'

    def dropout_selu_impl(x, rate, alpha, noise_shape, seed, name):
        if False:
            print('Hello World!')
        keep_prob = 1.0 - rate
        x = ops.convert_to_tensor(x, name='x')
        if isinstance(keep_prob, numbers.Real) and (not 0 < keep_prob <= 1):
            raise ValueError('keep_prob must be a scalar tensor or a float in the range (0, 1], got %g' % keep_prob)
        keep_prob = ops.convert_to_tensor(keep_prob, dtype=x.dtype, name='keep_prob')
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        alpha = ops.convert_to_tensor(alpha, dtype=x.dtype, name='alpha')
        keep_prob.get_shape().assert_is_compatible_with(tensor_shape.scalar())
        if tensor_util.constant_value(keep_prob) == 1:
            return x
        noise_shape = noise_shape if noise_shape is not None else array_ops.shape(x)
        random_tensor = keep_prob
        random_tensor += random_ops.random_uniform(noise_shape, seed=seed, dtype=x.dtype)
        binary_tensor = math_ops.floor(random_tensor)
        ret = x * binary_tensor + alpha * (1 - binary_tensor)
        a = tf.sqrt(fixedPointVar / (keep_prob * ((1 - keep_prob) * tf.pow(alpha - fixedPointMean, 2) + fixedPointVar)))
        b = fixedPointMean - a * (keep_prob * fixedPointMean + (1 - keep_prob) * alpha)
        ret = a * ret + b
        ret.set_shape(x.get_shape())
        return ret
    with ops.name_scope(name, 'dropout', [x]) as name:
        return utils.smart_cond(training, lambda : dropout_selu_impl(x, keep_prob, alpha, noise_shape, seed, name), lambda : array_ops.identity(x))
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
learning_rate = 0.001
training_epochs = 50
batch_size = 100
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, 10])
keep_prob = tf.placeholder(tf.float32)
W1 = tf.get_variable('W1', shape=[784, 512], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([512]))
L1 = selu(tf.matmul(X, W1) + b1)
L1 = dropout_selu(L1, keep_prob=keep_prob)
W2 = tf.get_variable('W2', shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([512]))
L2 = selu(tf.matmul(L1, W2) + b2)
L2 = dropout_selu(L2, keep_prob=keep_prob)
W3 = tf.get_variable('W3', shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([512]))
L3 = selu(tf.matmul(L2, W3) + b3)
L3 = dropout_selu(L3, keep_prob=keep_prob)
W4 = tf.get_variable('W4', shape=[512, 512], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([512]))
L4 = selu(tf.matmul(L3, W4) + b4)
L4 = dropout_selu(L4, keep_prob=keep_prob)
W5 = tf.get_variable('W5', shape=[512, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L4, W5) + b5
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range(total_batch):
        (batch_xs, batch_ys) = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob: 0.7}
        (c, _) = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
print('Learning Finished!')
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={X: mnist.test.images, Y: mnist.test.labels, keep_prob: 1}))
r = random.randint(0, mnist.test.num_examples - 1)
print('Label: ', sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print('Prediction: ', sess.run(tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1], keep_prob: 1}))
'\nEpoch: 0001 cost = 0.447322626\nEpoch: 0002 cost = 0.157285590\nEpoch: 0003 cost = 0.121884535\nEpoch: 0004 cost = 0.098128681\nEpoch: 0005 cost = 0.082901778\nEpoch: 0006 cost = 0.075337573\nEpoch: 0007 cost = 0.069752543\nEpoch: 0008 cost = 0.060884363\nEpoch: 0009 cost = 0.055276413\nEpoch: 0010 cost = 0.054631256\nEpoch: 0011 cost = 0.049675195\nEpoch: 0012 cost = 0.049125314\nEpoch: 0013 cost = 0.047231930\nEpoch: 0014 cost = 0.041290121\nEpoch: 0015 cost = 0.043621063\nLearning Finished!\nAccuracy: 0.9804\n'