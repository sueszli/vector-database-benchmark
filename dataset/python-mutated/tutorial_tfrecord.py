"""You will learn.

1. How to save data into TFRecord format file.
2. How to read data from TFRecord format file.

Reference:
-----------
English : https://www.tensorflow.org/alpha/tutorials/load_data/images#build_a_tfdatadataset
          https://www.tensorflow.org/alpha/tutorials/load_data/tf_records#tfrecord_files_using_tfdata
Chinese : http://blog.csdn.net/u012759136/article/details/52232266
          https://github.com/ycszen/tf_lab/blob/master/reading_data/TensorFlow高效加载数据的方法.md

More
------
1. tutorial_tfrecord2.py
2. tutorial_cifar10_tfrecord.py

"""
import os
import numpy as np
import tensorflow as tf
from PIL import Image
import tensorlayer as tl
classes = ['/data/cat', '/data/dog']
cwd = os.getcwd()
writer = tf.io.TFRecordWriter('train.tfrecords')
for (index, name) in enumerate(classes):
    class_path = cwd + name + '/'
    for img_name in os.listdir(class_path):
        img_path = class_path + img_name
        img = Image.open(img_path)
        img = img.resize((224, 224))
        img_raw = img.tobytes()
        example = tf.train.Example(features=tf.train.Features(feature={'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[index])), 'img_raw': tf.train.Feature(bytes_list=tf.train.BytesList(value=[img_raw]))}))
        writer.write(example.SerializeToString())
writer.close()
raw_dataset = tf.data.TFRecordDataset('train.tfrecords')
for serialized_example in raw_dataset:
    example = tf.train.Example()
    example.ParseFromString(serialized_example.numpy())
    img_raw = example.features.feature['img_raw'].bytes_list.value
    label = example.features.feature['label'].int64_list.value
    image = Image.frombytes('RGB', (224, 224), img_raw[0])
    print(label)

def read_and_decode(filename):
    if False:
        while True:
            i = 10
    raw_dataset = tf.data.TFRecordDataset([filename]).shuffle(1000).batch(4)
    features = {}
    for serialized_example in raw_dataset:
        features['label'] = tf.io.FixedLenFeature([], tf.int64)
        features['img_raw'] = tf.io.FixedLenFeature([], tf.string)
        features = tf.io.parse_example(serialized_example, features)
        img_batch = tf.io.decode_raw(features['img_raw'], tf.uint8)
        img_batch = tf.reshape(img_batch, [4, 224, 224, 3])
        label_batch = tf.cast(features['label'], tf.int32)
        yield (img_batch, label_batch)
(img_batch, label_batch) = next(read_and_decode('train.tfrecords'))
print('img_batch   : %s' % img_batch.shape)
print('label_batch : %s' % label_batch.shape)
tl.visualize.images2d(img_batch, second=1, saveable=False, name='batch', dtype=None, fig_idx=2020121)