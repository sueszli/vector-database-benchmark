"""Tests for data_provider."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
from absl.testing import parameterized
import numpy as np
import tensorflow as tf
import data_provider

class DataProviderTest(tf.test.TestCase, parameterized.TestCase):

    @parameterized.named_parameters(('train', 'train'), ('validation', 'validation'))
    def test_data_provider(self, split_name):
        if False:
            print('Hello World!')
        dataset_dir = os.path.join(flags.FLAGS.test_srcdir, 'google3/third_party/tensorflow_models/gan/image_compression/testdata/')
        batch_size = 3
        patch_size = 8
        images = data_provider.provide_data(split_name, batch_size, dataset_dir, patch_size=8)
        self.assertListEqual([batch_size, patch_size, patch_size, 3], images.shape.as_list())
        with self.test_session(use_gpu=True) as sess:
            with tf.contrib.slim.queues.QueueRunners(sess):
                images_out = sess.run(images)
                self.assertEqual((batch_size, patch_size, patch_size, 3), images_out.shape)
                self.assertTrue(np.all(np.abs(images_out) <= 1.0))
if __name__ == '__main__':
    tf.test.main()