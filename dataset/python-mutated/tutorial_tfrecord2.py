"""You will learn.

1. How to convert CIFAR-10 dataset into TFRecord format file.
2. How to read CIFAR-10 from TFRecord format file.

More:
1. tutorial_tfrecord.py
2. tutoral_cifar10_tfrecord.py

"""
import os
import numpy as np
import tensorflow as tf
import tensorlayer as tl
(X_train, y_train, X_test, y_test) = tl.files.load_cifar10_dataset(shape=(-1, 32, 32, 3), plotable=False)
X_train = np.asarray(X_train, dtype=np.uint8)
y_train = np.asarray(y_train, dtype=np.int64)
X_test = np.asarray(X_test, dtype=np.float32)
y_test = np.asarray(y_test, dtype=np.int64)
print('X_train.shape', X_train.shape)
print('y_train.shape', y_train.shape)
print('X_test.shape', X_test.shape)
print('y_test.shape', y_test.shape)
print('X %s   y %s' % (X_test.dtype, y_test.dtype))
cwd = os.getcwd()
writer = tf.io.TFRecordWriter('train.cifar10')
for (index, img) in enumerate(X_train):
    img_raw = img.tobytes()
    label = int(y_train[index])
    example = tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])), 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
    writer.write(example.SerializeToString())
writer.close()

def read_and_decode(filename):
    if False:
        return 10
    batchsize = 4
    raw_dataset = tf.data.TFRecordDataset([filename]).shuffle(1000).batch(batchsize)
    features = {}
    for serialized_example in raw_dataset:
        features['label'] = tf.io.FixedLenFeature([], tf.int64)
        features['img_raw'] = tf.io.FixedLenFeature([], tf.string)
        features = tf.io.parse_example(serialized_example, features)
        img_batch = tf.io.decode_raw(features['img_raw'], tf.uint8)
        img_batch = tf.reshape(img_batch, [-1, 32, 32, 3])
        label_batch = tf.cast(features['label'], tf.int32)
        yield (img_batch, label_batch)
(img_batch, label_batch) = next(read_and_decode('train.tfrecords'))
print('img_batch   : %s' % img_batch.shape)
print('label_batch : %s' % label_batch.shape)
i = 0
for (img_batch, label_batch) in read_and_decode('train.cifar10'):
    tl.visualize.images2d(img_batch, second=1, saveable=False, name='batch' + str(i), dtype=np.uint8, fig_idx=2020121)
    i += 1
    if i >= 3:
        break