""" Neural Network with Eager API.

A 2-Hidden Layers Fully Connected Neural Network (a.k.a Multilayer Perceptron)
implementation with TensorFlow's Eager API. This example is using the MNIST database
of handwritten digits (http://yann.lecun.com/exdb/mnist/).

This example is using TensorFlow layers, see 'neural_network_raw' example for
a raw implementation with variables.

Links:
    [MNIST Dataset](http://yann.lecun.com/exdb/mnist/).

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
tf.enable_eager_execution()
tfe = tf.contrib.eager
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=False)
learning_rate = 0.001
num_steps = 1000
batch_size = 128
display_step = 100
n_hidden_1 = 256
n_hidden_2 = 256
num_input = 784
num_classes = 10
dataset = tf.data.Dataset.from_tensor_slices((mnist.train.images, mnist.train.labels))
dataset = dataset.repeat().batch(batch_size).prefetch(batch_size)
dataset_iter = tfe.Iterator(dataset)

class NeuralNet(tfe.Network):

    def __init__(self):
        if False:
            while True:
                i = 10
        super(NeuralNet, self).__init__()
        self.layer1 = self.track_layer(tf.layers.Dense(n_hidden_1, activation=tf.nn.relu))
        self.layer2 = self.track_layer(tf.layers.Dense(n_hidden_2, activation=tf.nn.relu))
        self.out_layer = self.track_layer(tf.layers.Dense(num_classes))

    def call(self, x):
        if False:
            for i in range(10):
                print('nop')
        x = self.layer1(x)
        x = self.layer2(x)
        return self.out_layer(x)
neural_net = NeuralNet()

def loss_fn(inference_fn, inputs, labels):
    if False:
        for i in range(10):
            print('nop')
    return tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=inference_fn(inputs), labels=labels))

def accuracy_fn(inference_fn, inputs, labels):
    if False:
        return 10
    prediction = tf.nn.softmax(inference_fn(inputs))
    correct_pred = tf.equal(tf.argmax(prediction, 1), labels)
    return tf.reduce_mean(tf.cast(correct_pred, tf.float32))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
grad = tfe.implicit_gradients(loss_fn)
average_loss = 0.0
average_acc = 0.0
for step in range(num_steps):
    d = dataset_iter.next()
    x_batch = d[0]
    y_batch = tf.cast(d[1], dtype=tf.int64)
    batch_loss = loss_fn(neural_net, x_batch, y_batch)
    average_loss += batch_loss
    batch_accuracy = accuracy_fn(neural_net, x_batch, y_batch)
    average_acc += batch_accuracy
    if step == 0:
        print('Initial loss= {:.9f}'.format(average_loss))
    optimizer.apply_gradients(grad(neural_net, x_batch, y_batch))
    if (step + 1) % display_step == 0 or step == 0:
        if step > 0:
            average_loss /= display_step
            average_acc /= display_step
        print('Step:', '%04d' % (step + 1), ' loss=', '{:.9f}'.format(average_loss), ' accuracy=', '{:.4f}'.format(average_acc))
        average_loss = 0.0
        average_acc = 0.0
testX = mnist.test.images
testY = mnist.test.labels
test_acc = accuracy_fn(neural_net, testX, testY)
print('Testset Accuracy: {:.4f}'.format(test_acc))