"""Tests for tf_record_creation_util.py."""
import os
import contextlib2
import tensorflow as tf
from object_detection.dataset_tools import tf_record_creation_util

class OpenOutputTfrecordsTests(tf.test.TestCase):

    def test_sharded_tfrecord_writes(self):
        if False:
            print('Hello World!')
        with contextlib2.ExitStack() as tf_record_close_stack:
            output_tfrecords = tf_record_creation_util.open_sharded_output_tfrecords(tf_record_close_stack, os.path.join(tf.test.get_temp_dir(), 'test.tfrec'), 10)
            for idx in range(10):
                output_tfrecords[idx].write('test_{}'.format(idx))
        for idx in range(10):
            tf_record_path = '{}-{:05d}-of-00010'.format(os.path.join(tf.test.get_temp_dir(), 'test.tfrec'), idx)
            records = list(tf.python_io.tf_record_iterator(tf_record_path))
            self.assertAllEqual(records, ['test_{}'.format(idx)])
if __name__ == '__main__':
    tf.test.main()