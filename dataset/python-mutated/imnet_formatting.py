"""LSUN dataset formatting.

Download and format the Imagenet dataset as follow:
mkdir [IMAGENET_PATH]
cd [IMAGENET_PATH]
for FILENAME in train_32x32.tar valid_32x32.tar train_64x64.tar valid_64x64.tar
do
    curl -O http://image-net.org/small/$FILENAME
    tar -xvf $FILENAME
done

Then use the script as follow:
for DIRNAME in train_32x32 valid_32x32 train_64x64 valid_64x64
do
    python imnet_formatting.py \\
        --file_out $DIRNAME \\
        --fn_root $DIRNAME
done

"""
from __future__ import print_function
import os
import os.path
import scipy.io
import scipy.io.wavfile
import scipy.ndimage
import tensorflow as tf
tf.flags.DEFINE_string('file_out', '', 'Filename of the output .tfrecords file.')
tf.flags.DEFINE_string('fn_root', '', 'Name of root file path.')
FLAGS = tf.flags.FLAGS

def _int64_feature(value):
    if False:
        i = 10
        return i + 15
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if False:
        return 10
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():
    if False:
        return 10
    'Main converter function.'
    fn_root = FLAGS.fn_root
    img_fn_list = os.listdir(fn_root)
    img_fn_list = [img_fn for img_fn in img_fn_list if img_fn.endswith('.png')]
    num_examples = len(img_fn_list)
    n_examples_per_file = 10000
    for (example_idx, img_fn) in enumerate(img_fn_list):
        if example_idx % n_examples_per_file == 0:
            file_out = '%s_%05d.tfrecords'
            file_out = file_out % (FLAGS.file_out, example_idx // n_examples_per_file)
            print('Writing on:', file_out)
            writer = tf.python_io.TFRecordWriter(file_out)
        if example_idx % 1000 == 0:
            print(example_idx, '/', num_examples)
        image_raw = scipy.ndimage.imread(os.path.join(fn_root, img_fn))
        rows = image_raw.shape[0]
        cols = image_raw.shape[1]
        depth = image_raw.shape[2]
        image_raw = image_raw.astype('uint8')
        image_raw = image_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={'height': _int64_feature(rows), 'width': _int64_feature(cols), 'depth': _int64_feature(depth), 'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
        if example_idx % n_examples_per_file == n_examples_per_file - 1:
            writer.close()
    writer.close()
if __name__ == '__main__':
    main()