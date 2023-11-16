""" Build an Image Dataset in TensorFlow.

For this example, you need to make your own set of images (JPEG).
We will show 2 different ways to build that dataset:

- From a root folder, that will have a sub-folder containing images for each class
    ```
    ROOT_FOLDER
       |-------- SUBFOLDER (CLASS 0)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
       |             
       |-------- SUBFOLDER (CLASS 1)
       |             |
       |             | ----- image1.jpg
       |             | ----- image2.jpg
       |             | ----- etc...
    ```

- From a plain text file, that will list all images with their class ID:
    ```
    /path/to/image/1.jpg CLASS_ID
    /path/to/image/2.jpg CLASS_ID
    /path/to/image/3.jpg CLASS_ID
    /path/to/image/4.jpg CLASS_ID
    etc...
    ```

Below, there are some parameters that you need to change (Marked 'CHANGE HERE'), 
such as the dataset path.

Author: Aymeric Damien
Project: https://github.com/aymericdamien/TensorFlow-Examples/
"""
from __future__ import print_function
import tensorflow as tf
import os
MODE = 'folder'
DATASET_PATH = '/path/to/dataset/'
N_CLASSES = 2
IMG_HEIGHT = 64
IMG_WIDTH = 64
CHANNELS = 3

def read_images(dataset_path, mode, batch_size):
    if False:
        for i in range(10):
            print('nop')
    (imagepaths, labels) = (list(), list())
    if mode == 'file':
        with open(dataset_path) as f:
            data = f.read().splitlines()
        for d in data:
            imagepaths.append(d.split(' ')[0])
            labels.append(int(d.split(' ')[1]))
    elif mode == 'folder':
        label = 0
        try:
            classes = sorted(os.walk(dataset_path).next()[1])
        except Exception:
            classes = sorted(os.walk(dataset_path).__next__()[1])
        for c in classes:
            c_dir = os.path.join(dataset_path, c)
            try:
                walk = os.walk(c_dir).next()
            except Exception:
                walk = os.walk(c_dir).__next__()
            for sample in walk[2]:
                if sample.endswith('.jpg') or sample.endswith('.jpeg'):
                    imagepaths.append(os.path.join(c_dir, sample))
                    labels.append(label)
            label += 1
    else:
        raise Exception('Unknown mode.')
    imagepaths = tf.convert_to_tensor(imagepaths, dtype=tf.string)
    labels = tf.convert_to_tensor(labels, dtype=tf.int32)
    (image, label) = tf.train.slice_input_producer([imagepaths, labels], shuffle=True)
    image = tf.read_file(image)
    image = tf.image.decode_jpeg(image, channels=CHANNELS)
    image = tf.image.resize_images(image, [IMG_HEIGHT, IMG_WIDTH])
    image = image * 1.0 / 127.5 - 1.0
    (X, Y) = tf.train.batch([image, label], batch_size=batch_size, capacity=batch_size * 8, num_threads=4)
    return (X, Y)
learning_rate = 0.001
num_steps = 10000
batch_size = 128
display_step = 100
dropout = 0.75
(X, Y) = read_images(DATASET_PATH, MODE, batch_size)

def conv_net(x, n_classes, dropout, reuse, is_training):
    if False:
        print('Hello World!')
    with tf.variable_scope('ConvNet', reuse=reuse):
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
logits_train = conv_net(X, N_CLASSES, dropout, reuse=False, is_training=True)
logits_test = conv_net(X, N_CLASSES, dropout, reuse=True, is_training=False)
loss_op = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_train, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)
correct_pred = tf.equal(tf.argmax(logits_test, 1), tf.cast(Y, tf.int64))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
init = tf.global_variables_initializer()
saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run(init)
    tf.train.start_queue_runners()
    for step in range(1, num_steps + 1):
        if step % display_step == 0:
            (_, loss, acc) = sess.run([train_op, loss_op, accuracy])
            print('Step ' + str(step) + ', Minibatch Loss= ' + '{:.4f}'.format(loss) + ', Training Accuracy= ' + '{:.3f}'.format(acc))
        else:
            sess.run(train_op)
    print('Optimization Finished!')
    saver.save(sess, 'my_tf_model')