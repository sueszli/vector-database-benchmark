""" Multi-GPU Training Example.

Train a convolutional neural network on multiple GPU with TensorFlow.

This example is using TensorFlow layers, see 'convolutional_network_raw' example
for a raw TensorFlow implementation with variables.

This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import division, print_function, absolute_import
import numpy as np
import tensorflow as tf
import time
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
num_gpus = 2
num_steps = 200
learning_rate = 0.001
batch_size = 1024
display_step = 10
num_input = 784
num_classes = 10
dropout = 0.75

def conv_net(x, n_classes, dropout, reuse, is_training):
    if False:
        i = 10
        return i + 15
    with tf.variable_scope('ConvNet', reuse=reuse):
        x = tf.reshape(x, shape=[-1, 28, 28, 1])
        x = tf.layers.conv2d(x, 64, 5, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.layers.conv2d(x, 256, 3, activation=tf.nn.relu)
        x = tf.layers.conv2d(x, 512, 3, activation=tf.nn.relu)
        x = tf.layers.max_pooling2d(x, 2, 2)
        x = tf.contrib.layers.flatten(x)
        x = tf.layers.dense(x, 2048)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)
        x = tf.layers.dense(x, 1024)
        x = tf.layers.dropout(x, rate=dropout, training=is_training)
        out = tf.layers.dense(x, n_classes)
        out = tf.nn.softmax(out) if not is_training else out
    return out

def average_gradients(tower_grads):
    if False:
        return 10
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        grads = []
        for (g, _) in grad_and_vars:
            expanded_g = tf.expand_dims(g, 0)
            grads.append(expanded_g)
        grad = tf.concat(grads, 0)
        grad = tf.reduce_mean(grad, 0)
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
PS_OPS = ['Variable', 'VariableV2', 'AutoReloadVariable']

def assign_to_device(device, ps_device='/cpu:0'):
    if False:
        while True:
            i = 10

    def _assign(op):
        if False:
            print('Hello World!')
        node_def = op if isinstance(op, tf.NodeDef) else op.node_def
        if node_def.op in PS_OPS:
            return '/' + ps_device
        else:
            return device
    return _assign
with tf.device('/cpu:0'):
    tower_grads = []
    reuse_vars = False
    X = tf.placeholder(tf.float32, [None, num_input])
    Y = tf.placeholder(tf.float32, [None, num_classes])
    for i in range(num_gpus):
        with tf.device(assign_to_device('/gpu:{}'.format(i), ps_device='/cpu:0')):
            _x = X[i * batch_size:(i + 1) * batch_size]
            _y = Y[i * batch_size:(i + 1) * batch_size]
            logits_train = conv_net(_x, num_classes, dropout, reuse=reuse_vars, is_training=True)
            logits_test = conv_net(_x, num_classes, dropout, reuse=True, is_training=False)
            loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_train, labels=_y))
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            grads = optimizer.compute_gradients(loss_op)
            if i == 0:
                correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.argmax(_y, 1))
                accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            reuse_vars = True
            tower_grads.append(grads)
    tower_grads = average_gradients(tower_grads)
    train_op = optimizer.apply_gradients(tower_grads)
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        sess.run(init)
        for step in range(1, num_steps + 1):
            (batch_x, batch_y) = mnist.train.next_batch(batch_size * num_gpus)
            ts = time.time()
            sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
            te = time.time() - ts
            if step % display_step == 0 or step == 1:
                (loss, acc) = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
                print('Step ' + str(step) + ': Minibatch Loss= ' + '{:.4f}'.format(loss) + ', Training Accuracy= ' + '{:.3f}'.format(acc) + ', %i Examples/sec' % int(len(batch_x) / te))
            step += 1
        print('Optimization Finished!')
        print('Testing Accuracy:', np.mean([sess.run(accuracy, feed_dict={X: mnist.test.images[i:i + batch_size], Y: mnist.test.labels[i:i + batch_size]}) for i in range(0, len(mnist.test.images), batch_size)]))