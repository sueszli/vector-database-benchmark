"""Tests for dataset_builder."""
import os
import numpy as np
import tensorflow as tf
from google.protobuf import text_format
from object_detection.builders import dataset_builder
from object_detection.core import standard_fields as fields
from object_detection.protos import input_reader_pb2
from object_detection.utils import dataset_util

class DatasetBuilderTest(tf.test.TestCase):

    def create_tf_record(self, has_additional_channels=False, num_examples=1):
        if False:
            print('Hello World!')
        path = os.path.join(self.get_temp_dir(), 'tfrecord')
        writer = tf.python_io.TFRecordWriter(path)
        image_tensor = np.random.randint(255, size=(4, 5, 3)).astype(np.uint8)
        additional_channels_tensor = np.random.randint(255, size=(4, 5, 1)).astype(np.uint8)
        flat_mask = 4 * 5 * [1.0]
        with self.test_session():
            encoded_jpeg = tf.image.encode_jpeg(tf.constant(image_tensor)).eval()
            encoded_additional_channels_jpeg = tf.image.encode_jpeg(tf.constant(additional_channels_tensor)).eval()
            for i in range(num_examples):
                features = {'image/source_id': dataset_util.bytes_feature(str(i)), 'image/encoded': dataset_util.bytes_feature(encoded_jpeg), 'image/format': dataset_util.bytes_feature('jpeg'.encode('utf8')), 'image/height': dataset_util.int64_feature(4), 'image/width': dataset_util.int64_feature(5), 'image/object/bbox/xmin': dataset_util.float_list_feature([0.0]), 'image/object/bbox/xmax': dataset_util.float_list_feature([1.0]), 'image/object/bbox/ymin': dataset_util.float_list_feature([0.0]), 'image/object/bbox/ymax': dataset_util.float_list_feature([1.0]), 'image/object/class/label': dataset_util.int64_list_feature([2]), 'image/object/mask': dataset_util.float_list_feature(flat_mask)}
                if has_additional_channels:
                    additional_channels_key = 'image/additional_channels/encoded'
                    features[additional_channels_key] = dataset_util.bytes_list_feature([encoded_additional_channels_jpeg] * 2)
                example = tf.train.Example(features=tf.train.Features(feature=features))
                writer.write(example.SerializeToString())
            writer.close()
        return path

    def test_build_tf_record_input_reader(self):
        if False:
            return 10
        tf_record_path = self.create_tf_record()
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        tensor_dict = dataset_builder.make_initializable_iterator(dataset_builder.build(input_reader_proto, batch_size=1)).get_next()
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
        self.assertTrue(fields.InputDataFields.groundtruth_instance_masks not in output_dict)
        self.assertEquals((1, 4, 5, 3), output_dict[fields.InputDataFields.image].shape)
        self.assertAllEqual([[2]], output_dict[fields.InputDataFields.groundtruth_classes])
        self.assertEquals((1, 1, 4), output_dict[fields.InputDataFields.groundtruth_boxes].shape)
        self.assertAllEqual([0.0, 0.0, 1.0, 1.0], output_dict[fields.InputDataFields.groundtruth_boxes][0][0])

    def test_build_tf_record_input_reader_and_load_instance_masks(self):
        if False:
            i = 10
            return i + 15
        tf_record_path = self.create_tf_record()
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      load_instance_masks: true\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        tensor_dict = dataset_builder.make_initializable_iterator(dataset_builder.build(input_reader_proto, batch_size=1)).get_next()
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
        self.assertAllEqual((1, 1, 4, 5), output_dict[fields.InputDataFields.groundtruth_instance_masks].shape)

    def test_build_tf_record_input_reader_with_batch_size_two(self):
        if False:
            i = 10
            return i + 15
        tf_record_path = self.create_tf_record()
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)

        def one_hot_class_encoding_fn(tensor_dict):
            if False:
                while True:
                    i = 10
            tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.one_hot(tensor_dict[fields.InputDataFields.groundtruth_classes] - 1, depth=3)
            return tensor_dict
        tensor_dict = dataset_builder.make_initializable_iterator(dataset_builder.build(input_reader_proto, transform_input_data_fn=one_hot_class_encoding_fn, batch_size=2)).get_next()
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
        self.assertAllEqual([2, 4, 5, 3], output_dict[fields.InputDataFields.image].shape)
        self.assertAllEqual([2, 1, 3], output_dict[fields.InputDataFields.groundtruth_classes].shape)
        self.assertAllEqual([2, 1, 4], output_dict[fields.InputDataFields.groundtruth_boxes].shape)
        self.assertAllEqual([[[0.0, 0.0, 1.0, 1.0]], [[0.0, 0.0, 1.0, 1.0]]], output_dict[fields.InputDataFields.groundtruth_boxes])

    def test_build_tf_record_input_reader_with_batch_size_two_and_masks(self):
        if False:
            for i in range(10):
                print('nop')
        tf_record_path = self.create_tf_record()
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      load_instance_masks: true\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)

        def one_hot_class_encoding_fn(tensor_dict):
            if False:
                i = 10
                return i + 15
            tensor_dict[fields.InputDataFields.groundtruth_classes] = tf.one_hot(tensor_dict[fields.InputDataFields.groundtruth_classes] - 1, depth=3)
            return tensor_dict
        tensor_dict = dataset_builder.make_initializable_iterator(dataset_builder.build(input_reader_proto, transform_input_data_fn=one_hot_class_encoding_fn, batch_size=2)).get_next()
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
        self.assertAllEqual([2, 1, 4, 5], output_dict[fields.InputDataFields.groundtruth_instance_masks].shape)

    def test_raises_error_with_no_input_paths(self):
        if False:
            print('Hello World!')
        input_reader_text_proto = '\n      shuffle: false\n      num_readers: 1\n      load_instance_masks: true\n    '
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        with self.assertRaises(ValueError):
            dataset_builder.build(input_reader_proto, batch_size=1)

    def test_sample_all_data(self):
        if False:
            i = 10
            return i + 15
        tf_record_path = self.create_tf_record(num_examples=2)
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      sample_1_of_n_examples: 1\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        tensor_dict = dataset_builder.make_initializable_iterator(dataset_builder.build(input_reader_proto, batch_size=1)).get_next()
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
            self.assertAllEqual(['0'], output_dict[fields.InputDataFields.source_id])
            output_dict = sess.run(tensor_dict)
            self.assertEquals(['1'], output_dict[fields.InputDataFields.source_id])

    def test_sample_one_of_n_shards(self):
        if False:
            while True:
                i = 10
        tf_record_path = self.create_tf_record(num_examples=4)
        input_reader_text_proto = "\n      shuffle: false\n      num_readers: 1\n      sample_1_of_n_examples: 2\n      tf_record_input_reader {{\n        input_path: '{0}'\n      }}\n    ".format(tf_record_path)
        input_reader_proto = input_reader_pb2.InputReader()
        text_format.Merge(input_reader_text_proto, input_reader_proto)
        tensor_dict = dataset_builder.make_initializable_iterator(dataset_builder.build(input_reader_proto, batch_size=1)).get_next()
        with tf.train.MonitoredSession() as sess:
            output_dict = sess.run(tensor_dict)
            self.assertAllEqual(['0'], output_dict[fields.InputDataFields.source_id])
            output_dict = sess.run(tensor_dict)
            self.assertEquals(['2'], output_dict[fields.InputDataFields.source_id])

class ReadDatasetTest(tf.test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        self._path_template = os.path.join(self.get_temp_dir(), 'examples_%s.txt')
        for i in range(5):
            path = self._path_template % i
            with tf.gfile.Open(path, 'wb') as f:
                f.write('\n'.join([str(i + 1), str((i + 1) * 10)]))
        self._shuffle_path_template = os.path.join(self.get_temp_dir(), 'shuffle_%s.txt')
        for i in range(2):
            path = self._shuffle_path_template % i
            with tf.gfile.Open(path, 'wb') as f:
                f.write('\n'.join([str(i)] * 5))

    def _get_dataset_next(self, files, config, batch_size):
        if False:
            while True:
                i = 10

        def decode_func(value):
            if False:
                while True:
                    i = 10
            return [tf.string_to_number(value, out_type=tf.int32)]
        dataset = dataset_builder.read_dataset(tf.data.TextLineDataset, files, config)
        dataset = dataset.map(decode_func)
        dataset = dataset.batch(batch_size)
        return dataset.make_one_shot_iterator().get_next()

    def test_make_initializable_iterator_with_hashTable(self):
        if False:
            i = 10
            return i + 15
        keys = [1, 0, -1]
        dataset = tf.data.Dataset.from_tensor_slices([[1, 2, -1, 5]])
        table = tf.contrib.lookup.HashTable(initializer=tf.contrib.lookup.KeyValueTensorInitializer(keys=keys, values=list(reversed(keys))), default_value=100)
        dataset = dataset.map(table.lookup)
        data = dataset_builder.make_initializable_iterator(dataset).get_next()
        init = tf.tables_initializer()
        with self.test_session() as sess:
            sess.run(init)
            self.assertAllEqual(sess.run(data), [-1, 100, 1, 100])

    def test_read_dataset(self):
        if False:
            i = 10
            return i + 15
        config = input_reader_pb2.InputReader()
        config.num_readers = 1
        config.shuffle = False
        data = self._get_dataset_next([self._path_template % '*'], config, batch_size=20)
        with self.test_session() as sess:
            self.assertAllEqual(sess.run(data), [[1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3, 30, 4, 40, 5, 50]])

    def test_reduce_num_reader(self):
        if False:
            return 10
        config = input_reader_pb2.InputReader()
        config.num_readers = 10
        config.shuffle = False
        data = self._get_dataset_next([self._path_template % '*'], config, batch_size=20)
        with self.test_session() as sess:
            self.assertAllEqual(sess.run(data), [[1, 10, 2, 20, 3, 30, 4, 40, 5, 50, 1, 10, 2, 20, 3, 30, 4, 40, 5, 50]])

    def test_enable_shuffle(self):
        if False:
            print('Hello World!')
        config = input_reader_pb2.InputReader()
        config.num_readers = 1
        config.shuffle = True
        tf.set_random_seed(1)
        data = self._get_dataset_next([self._shuffle_path_template % '*'], config, batch_size=10)
        expected_non_shuffle_output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        with self.test_session() as sess:
            self.assertTrue(np.any(np.not_equal(sess.run(data), expected_non_shuffle_output)))

    def test_disable_shuffle_(self):
        if False:
            return 10
        config = input_reader_pb2.InputReader()
        config.num_readers = 1
        config.shuffle = False
        data = self._get_dataset_next([self._shuffle_path_template % '*'], config, batch_size=10)
        expected_non_shuffle_output = [0, 0, 0, 0, 0, 1, 1, 1, 1, 1]
        with self.test_session() as sess:
            self.assertAllEqual(sess.run(data), [expected_non_shuffle_output])

    def test_read_dataset_single_epoch(self):
        if False:
            while True:
                i = 10
        config = input_reader_pb2.InputReader()
        config.num_epochs = 1
        config.num_readers = 1
        config.shuffle = False
        data = self._get_dataset_next([self._path_template % '0'], config, batch_size=30)
        with self.test_session() as sess:
            self.assertAllEqual(sess.run(data), [[1, 10]])
            self.assertRaises(tf.errors.OutOfRangeError, sess.run, data)
if __name__ == '__main__':
    tf.test.main()