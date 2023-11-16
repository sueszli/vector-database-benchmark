""" TensorFlow Dataset API.

In this example, we will show how to load numpy array data into the new 
TensorFlow 'Dataset' API. The Dataset API implements an optimized data pipeline
with queues, that make data processing and training faster (especially on GPU).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
learning_rate = 0.001
num_steps = 2000
batch_size = 128
display_step = 100
n_input = 784
n_classes = 10
dropout = 0.75
sess = tf.Session()
dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels))
dataset = dataset.repeat()
dataset = dataset.batch(batch_size)
dataset = dataset.prefetch(batch_size)
iterator = dataset.make_initializable_iterator()
sess.run(iterator.initializer)
(X, Y) = iterator.get_next()

def conv_net(x, n_classes, dropout, reuse, is_training):
    if False:
        print('Hello World!')
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)
        fc1 = tf.contrib.layers.flatten(conv2)
        fc1 = tf.layers.dense(fc1, 1024)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)
        out = tf.layers.dense(fc1, n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out
logits_train = conv_net(X, n_classes, dropout, reuse=False, is_training=True)
logits_test = conv_net(X, n_classes, dropout, reuse=True, is_training=False)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
sess.run(init)
for step in range(1, num_steps + 1):
    sess.run(train_op)
    if step % display_step == 0 or step == 1:
        (loss, acc) = sess.run([loss_op, accuracy])
        print('Step ' + str(step) + ', Minibatch Loss= ' + '{:.4f}'.format(loss) + ', Training Accuracy= ' + '{:.3f}'.format(acc))
print('Optimization Finished!')