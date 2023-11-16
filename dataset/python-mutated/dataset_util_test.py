"""Tests for object_detection.utils.dataset_util."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import tensorflow as tf
from object_detection.utils import dataset_util

class DatasetUtilTest(tf.test.TestCase):

    def test_read_examples_list(self):
        if False:
            print('Hello World!')
        example_list_data = 'example1 1\nexample2 2'
        example_list_path = os.path.join(self.get_temp_dir(), 'examples.txt')
        with tf.gfile.Open(example_list_path, 'wb') as f:
            f.write(example_list_data)
        examples = dataset_util.read_examples_list(example_list_path)
        self.assertListEqual(['example1', 'example2'], examples)
if __name__ == '__main__':
    tf.test.main()