"""Tests for input_reader_builder."""
import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import input_reader_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util

class InputReaderBuilderTest(tf.test.TestCase):

    def create_tf_record(self):
        if False:
            for i in range(10):
                print('nop')
        path = os.path.join(self.get_temp_dir(), 'tfrecord')
        writer = tf.python_io.TFRecordWriter(path)
        image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
        flat_mask = 4 * 5 * [1.0]
        with self.test_session():
            encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
        example = tf.train.Example(features=tf.train.Features(feature={'image/encoded': dataset_util.bytes_feature(encoded_jpeg), 'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')), 'image/height': dataset_util.int64_feature(4), 'image/width': dataset_util.int64_feature(5), 'image/object/bbox/xmin': dataset_util.float_list_feature([0.0]), 'image/object/bbox/xmax': dataset_util.float_list_feature([1.0]), 'image/object/bbox/ymin': dataset_util.float_list_feature([0.0]), 'image/object/bbox/ymax': dataset_util.float_list_feature([1.0]), 'image/object/class/label': dataset_util.int64_list_feature([2]), 'image/object/mask': dataset_util.float_list_feature(flat_mask)}))
        writer.write(example.SerializeToString())
        writer.close()
        return path

    def test_build_tf_record_input_reader(self):
        if False:
            while True:
                i = 10
        tf_record_path = self.create_tf_record()
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        tensor_dict = input_reader_builder.build(input_reader_proto)
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
        self.assertTrue(fields.InputDataFields.groundtruth_instance_masks not in output_dict)
        self.assertEquals((4, 5, 3), output_dict[fields.InputDataFields.image].shape)
        self.assertEquals([2], output_dict[fields.InputDataFields.groundtruth_classes])
        self.assertEquals((1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
        self.assertAllEqual([0.0, 0.0, 1.0, 1.0], output_dict[fields.InputDataFields.groundtruth_boxes][0])

    def test_build_tf_record_input_reader_and_load_instance_masks(self):
        if False:
            for i in range(10):
                print('nop')
        tf_record_path = self.create_tf_record()
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      load_instance_masks: true\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        tensor_dict = input_reader_builder.build(input_reader_proto)
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
        self.assertEquals((4, 5, 3), output_dict[fields.InputDataFields.image].shape)
        self.assertEquals([2], output_dict[fields.InputDataFields.groundtruth_classes])
        self.assertEquals((1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
        self.assertAllEqual([0.0, 0.0, 1.0, 1.0], output_dict[fields.InputDataFields.groundtruth_boxes][0])
        self.assertAllEqual((1, 4, 5), output_dict[fields.InputDataFields.groundtruth_instance_masks].shape)

    def test_raises_error_with_no_input_paths(self):
        if False:
            while True:
                i = 10
        input_reader_text_proto = '\n      shuffle: false\n      num_readers: 1\n      load_instance_masks: true\n    '
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        with self.assertRaises(ValueError):
            input_reader_builder.build(input_reader_proto)
if __name__ == '__main__':
    tf.test.main()