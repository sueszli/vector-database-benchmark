""" Logistic Regression with Eager API.

A logistic regression learning algorithm example using TensorFlow's Eager API.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import absolute_import, division, print_function
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=False)
learning_rate = 0.1
batch_size = 128
num_steps = 1000
display_step = 100
dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels))
dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
dataset_iter = tfe.Iterator(dataset)
W = tfe.Variable(tf.zeros([784, 10]), name='weights')
b = tfe.Variable(tf.zeros([10]), name='bias')

def logistic_regression(inputs):
    if False:
        while True:
            i = 10
    return tf.matmul(inputs, W) + b

def loss_fn(inference_fn, inputs, labels):
    if False:
        for i in range(10):
            print('nop')
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=inference_fn(inputs), labels=labels))

def accuracy_fn(inference_fn, inputs, labels):
    if False:
        print('Hello World!')
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
grad = tfe.implicit_gradients(loss_fn)
average_loss = 0.0
average_acc = 0.0
for step in range(num_steps):
    d = dataset_iter.next()
    x_batch = d[0]
    y_batch = tf.cast(d[1], dtype=tf.int64)
    batch_loss = loss_fn(logistic_regression, x_batch, y_batch)
    average_loss += batch_loss
    batch_accuracy = accuracy_fn(logistic_regression, x_batch, y_batch)
    average_acc += batch_accuracy
    if step == 0:
        print('Initial loss= {:.9f}'.format(average_loss))
    optimizer.apply_gradients(grad(logistic_regression, x_batch, y_batch))
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print('Step:', '%04d' % (step + 1), ' loss=', '{:.9f}'.format(average_loss), ' accuracy=', '{:.4f}'.format(average_acc))
        average_loss = 0.0
        average_acc = 0.0
testX = mnist.test.images
testY = mnist.test.labels
test_acc = accuracy_fn(logistic_regression, testX, testY)
print('Testset Accuracy: {:.4f}'.format(test_acc))