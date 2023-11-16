"""LSUN dataset formatting.

Download and format the LSUN dataset as follow:
git clone https://github.com/fyu/lsun.git
cd lsun
python2.7 download.py -c [CATEGORY]

Then unzip the downloaded .zip files before executing:
python2.7 data.py export [IMAGE_DB_PATH] --out_dir [LSUN_FOLDER] --flat

Then use the script as follow:
python lsun_formatting.py \\
    --file_out [OUTPUT_FILE_PATH_PREFIX] \\
    --fn_root [LSUN_FOLDER]

"""
from __future__ import print_function
import os
import os.path
import numpy
import skimage.transform
from PIL import Image
import tensorflow as tf
tf.flags.DEFINE_string('file_out', '', 'Filename of the output .tfrecords file.')
tf.flags.DEFINE_string('fn_root', '', 'Name of root file path.')
FLAGS = tf.flags.FLAGS

def _int64_feature(value):
    if False:
        return 10
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _bytes_feature(value):
    if False:
        i = 10
        return i + 15
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def main():
    if False:
        i = 10
        return i + 15
    'Main converter function.'
    fn_root = FLAGS.fn_root
    img_fn_list = os.listdir(fn_root)
    img_fn_list = [img_fn for img_fn in img_fn_list if img_fn.endswith('.webp')]
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
        image_raw = numpy.array(Image.open(os.path.join(fn_root, img_fn)))
        rows = image_raw.shape[0]
        cols = image_raw.shape[1]
        depth = image_raw.shape[2]
        downscale = min(rows / 96.0, cols / 96.0)
        image_raw = skimage.transform.pyramid_reduce(image_raw, downscale)
        image_raw *= 255.0
        image_raw = image_raw.astype('uint8')
        rows = image_raw.shape[0]
        cols = image_raw.shape[1]
        depth = image_raw.shape[2]
        image_raw = image_raw.tostring()
        example = tf.train.Example(features=tf.train.Features(feature={'height': _int64_feature(rows), 'width': _int64_feature(cols), 'depth': _int64_feature(depth), 'image_raw': _bytes_feature(image_raw)}))
        writer.write(example.SerializeToString())
        if example_idx % n_examples_per_file == n_examples_per_file - 1:
            writer.close()
    writer.close()
if __name__ == '__main__':
    main()