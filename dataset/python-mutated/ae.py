from __future__ import division, print_function, absolute_import
import tensorflow.contrib.layers as lays
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from skimage import transform
from tensorflow.examples.tutorials.mnist import input_data
batch_size = 500
epoch_num = 5
lr = 0.001

def resize_batch(imgs):
    if False:
        for i in range(10):
            print('nop')
    imgs = imgs.reshape((-1, 28, 28, 1))
    resized_imgs = np.zeros((imgs.shape[0], 32, 32, 1))
    for i in range(imgs.shape[0]):
        resized_imgs[i, ..., 0] = transform.resize(imgs[i, ..., 0], (32, 32))
    return resized_imgs

def autoencoder(inputs):
    if False:
        while True:
            i = 10
    net = lays.conv2d(inputs, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 16, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d(net, 8, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 16, [5, 5], stride=4, padding='SAME')
    net = lays.conv2d_transpose(net, 32, [5, 5], stride=2, padding='SAME')
    net = lays.conv2d_transpose(net, 1, [5, 5], stride=2, padding='SAME', activation_fn=tf.nn.tanh)
    return net
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
batch_per_ep = mnist.train.num_examples // batch_size
ae_inputs = tf.placeholder(tf.float32, (None, 32, 32, 1))
ae_outputs = autoencoder(ae_inputs)
loss = tf.reduce_mean(tf.square(ae_outputs - ae_inputs))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
init = tf.global_variables_initializer()
with tf.Session() as sess:
    sess.run(init)
    for ep in range(epoch_num):
        for batch_n in range(batch_per_ep):
            (batch_img, batch_label) = mnist.train.next_batch(batch_size)
            batch_img = batch_img.reshape((-1, 28, 28, 1))
            batch_img = resize_batch(batch_img)
            (_, c) = sess.run([train_op, loss], feed_dict={ae_inputs: batch_img})
            print('Epoch: {} - cost= {:.5f}'.format(ep + 1, c))
    (batch_img, batch_label) = mnist.test.next_batch(50)
    batch_img = resize_batch(batch_img)
    recon_img = sess.run([ae_outputs], feed_dict={ae_inputs: batch_img})[0]
    plt.figure(1)
    plt.title('Reconstructed Images')
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(recon_img[i, ..., 0], cmap='gray')
    plt.figure(2)
    plt.title('Input Images')
    for i in range(50):
        plt.subplot(5, 10, i + 1)
        plt.imshow(batch_img[i, ..., 0], cmap='gray')
    plt.show()