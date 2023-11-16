""" Bi-directional Recurrent Neural Network.

A Bi-directional Recurrent Neural Network (LSTM) implementation example using 
TensorFlow library. This example is using the MNIST database of handwritten 
digits (http://yann.lecun.com/exdb/mnist/)

Links:
    [Long Short Term Memory](http://deeplearning.cs.cmu.edu/pdfs/Hochreiter97_lstm.pdf)
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.contrib import rnn
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
'\nTo classify images using a bidirectional recurrent neural network, we consider\nevery image row as a sequence of pixels. Because MNIST image shape is 28*28px,\nwe will then handle 28 sequences of 28 steps for every sample.\n'
learning_rate = 0.001
training_steps = 10000
batch_size = 128
display_step = 200
num_input = 28
timesteps = 28
num_hidden = 128
num_classes = 10
X = tf.placeholder('float', [None, timesteps, num_input])
Y = tf.placeholder('float', [None, num_classes])
weights = {'out': tf.Variable(tf.random_normal([2 * num_hidden, num_classes]))}
biases = {'out': tf.Variable(tf.random_normal([num_classes]))}

def BiRNN(x, weights, biases):
    if False:
        return 10
    x = tf.unstack(x, timesteps, 1)
    lstm_fw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    lstm_bw_cell = rnn.BasicLSTMCell(num_hidden, forget_bias=1.0)
    try:
        (outputs, _, _) = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    except Exception:
        outputs = rnn.static_bidirectional_rnn(lstm_fw_cell, lstm_bw_cell, x, dtype=tf.float32)
    return tf.matmul(outputs[-1], weights['out']) + biases['out']
logits = BiRNN(X, weights, biases)
prediction = tf.nn.softmax(logits)
loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for step in range(1, training_steps + 1):
        (batch_x, batch_y) = mnist.train.next_batch(batch_size)
        batch_x = batch_x.reshape((batch_size, timesteps, num_input))
        sess.run(train_op, feed_dict={X: batch_x, Y: batch_y})
        if step % display_step == 0 or step == 1:
            (loss, acc) = sess.run([loss_op, accuracy], feed_dict={X: batch_x, Y: batch_y})
            print('Step ' + str(step) + ', Minibatch Loss= ' + '{:.4f}'.format(loss) + ', Training Accuracy= ' + '{:.3f}'.format(acc))
    print('Optimization Finished!')
    test_len = 128
    test_data = mnist.test.images[:test_len].reshape((-1, timesteps, num_input))
    test_label = mnist.test.labels[:test_len]
    print('Testing Accuracy:', sess.run(accuracy, feed_dict={X: test_data, Y: test_label}))