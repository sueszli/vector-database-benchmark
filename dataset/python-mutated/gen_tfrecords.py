"""An example script to generate a tfrecord file from a folder containing the
renderings.

Example usage:
  python gen_tfrecords.py --input=FOLDER --output=output.tfrecord

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import numpy as np
import os
from scipy import misc
import tensorflow as tf
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('input', '', 'Input folder containing images')
tf.app.flags.DEFINE_string('output', '', 'Output tfrecord.')

def get_matrix(lines):
    if False:
        for i in range(10):
            print('nop')
    return np.array([[float(y) for y in x.strip().split(' ')] for x in lines])

def read_model_view_matrices(filename):
    if False:
        print('Hello World!')
    with open(filename, 'r') as f:
        lines = f.readlines()
    return (get_matrix(lines[:4]), get_matrix(lines[4:]))

def bytes_feature(values):
    if False:
        return 10
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[values]))

def generate():
    if False:
        return 10
    with tf.python_io.TFRecordWriter(FLAGS.output) as tfrecord_writer:
        with tf.Graph().as_default():
            im0 = tf.placeholder(dtype=tf.uint8)
            im1 = tf.placeholder(dtype=tf.uint8)
            encoded0 = tf.image.encode_png(im0)
            encoded1 = tf.image.encode_png(im1)
            with tf.Session() as sess:
                count = 0
                indir = FLAGS.input + '/'
                while tf.gfile.Exists(indir + '%06d.txt' % count):
                    print('saving %06d' % count)
                    image0 = misc.imread(indir + '%06d.png' % (count * 2))
                    image1 = misc.imread(indir + '%06d.png' % (count * 2 + 1))
                    (mat0, mat1) = read_model_view_matrices(indir + '%06d.txt' % count)
                    mati0 = np.linalg.inv(mat0).flatten()
                    mati1 = np.linalg.inv(mat1).flatten()
                    mat0 = mat0.flatten()
                    mat1 = mat1.flatten()
                    (st0, st1) = sess.run([encoded0, encoded1], feed_dict={im0: image0, im1: image1})
                    example = tf.train.Example(features=tf.train.Features(feature={'img0': bytes_feature(st0), 'img1': bytes_feature(st1), 'mv0': tf.train.Feature(float_list=tf.train.FloatList(value=mat0)), 'mvi0': tf.train.Feature(float_list=tf.train.FloatList(value=mati0)), 'mv1': tf.train.Feature(float_list=tf.train.FloatList(value=mat1)), 'mvi1': tf.train.Feature(float_list=tf.train.FloatList(value=mati1))}))
                    tfrecord_writer.write(example.SerializeToString())
                    count += 1

def main(argv):
    if False:
        print('Hello World!')
    del argv
    generate()
if __name__ == '__main__':
    tf.app.run()