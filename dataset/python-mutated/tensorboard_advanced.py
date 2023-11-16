"""
Graph and Loss visualization using Tensorboard.
This example is using the MNIST database of handwritten digits
(http://yann.lecun.com/exdb/mnist/)

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('/tmp/data/', one_hot=True)
learning_rate = 0.01
training_epochs = 25
batch_size = 100
display_step = 1
logs_path = '/tmp/tensorflow_logs/example/'
n_hidden_1 = 256
n_hidden_2 = 256
n_input = 784
n_classes = 10
x = tf.placeholder(tf.float32, [None, 784], name='InputData')
y = tf.placeholder(tf.float32, [None, 10], name='LabelData')

def multilayer_perceptron(x, weights, biases):
    if False:
        for i in range(10):
            print('nop')
    layer_1 = tf.add(tf.matmul(x, weights['w1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    tf.summary.histogram('relu1', layer_1)
    layer_2 = tf.add(tf.matmul(layer_1, weights['w2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    tf.summary.histogram('relu2', layer_2)
    out_layer = tf.add(tf.matmul(layer_2, weights['w3']), biases['b3'])
    return out_layer
weights = {'w1': tf.Variable(tf.random_normal([n_input, n_hidden_1]), name='W1'), 'w2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2]), name='W2'), 'w3': tf.Variable(tf.random_normal([n_hidden_2, n_classes]), name='W3')}
biases = {'b1': tf.Variable(tf.random_normal([n_hidden_1]), name='b1'), 'b2': tf.Variable(tf.random_normal([n_hidden_2]), name='b2'), 'b3': tf.Variable(tf.random_normal([n_classes]), name='b3')}
with tf.name_scope('Model'):
    pred = multilayer_perceptron(x, weights, biases)
with tf.name_scope('Loss'):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
with tf.name_scope('SGD'):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    grads = tf.gradients(loss, tf.trainable_variables())
    grads = list(zip(grads, tf.trainable_variables()))
    apply_grads = optimizer.apply_gradients(grads_and_vars=grads)
with tf.name_scope('Accuracy'):
    acc = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(acc, tf.float32))
init = tf.global_variables_initializer()
tf.summary.scalar('loss', loss)
tf.summary.scalar('accuracy', acc)
for var in tf.trainable_variables():
    tf.summary.histogram(var.name, var)
for (grad, var) in grads:
    tf.summary.histogram(var.name + '/gradient', grad)
merged_summary_op = tf.summary.merge_all()
with tf.Session() as sess:
    sess.run(init)
    summary_writer = tf.summary.FileWriter(logs_path, graph=tf.get_default_graph())
    for epoch in range(training_epochs):
        avg_cost = 0.0
        total_batch = int(mnist.train.num_examples / batch_size)
        for i in range(total_batch):
            (batch_xs, batch_ys) = mnist.train.next_batch(batch_size)
            (_, c, summary) = sess.run([apply_grads, loss, merged_summary_op], feed_dict={x: batch_xs, y: batch_ys})
            summary_writer.add_summary(summary, epoch * total_batch + i)
            avg_cost += c / total_batch
        if (epoch + 1) % display_step == 0:
            print('Epoch:', '%04d' % (epoch + 1), 'cost=', '{:.9f}'.format(avg_cost))
    print('Optimization Finished!')
    print('Accuracy:', acc.eval({x: mnist.test.images, y: mnist.test.labels}))
    print('Run the command line:\n--> tensorboard --logdir=/tmp/tensorflow_logs \nThen open http://0.0.0.0:6006/ into your web browser')