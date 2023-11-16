"""A binary to evaluate Inception on the ImageNet data set.

Note that using the supplied pre-trained inception checkpoint, the eval should
achieve:
  precision @ 1 = 0.7874 recall @ 5 = 0.9436 [50000 examples]

See the README.md for more details.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import tensorflow as tf
from inception import inception_eval
from inception.imagenet_data import ImagenetData
FLAGS = tf.app.flags.FLAGS

def main(unused_argv=None):
    if False:
        while True:
            i = 10
    dataset = ImagenetData(subset=FLAGS.subset)
    assert dataset.data_files()
    if tf.gfile.Exists(FLAGS.eval_dir):
        tf.gfile.DeleteRecursively(FLAGS.eval_dir)
    tf.gfile.MakeDirs(FLAGS.eval_dir)
    inception_eval.evaluate(dataset)
if __name__ == '__main__':
    tf.app.run()