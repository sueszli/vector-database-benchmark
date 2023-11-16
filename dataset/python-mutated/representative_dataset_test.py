"""Tests for representative_dataset.py."""
import random
import numpy as np
from tensorflow.compiler.mlir.quantization.tensorflow.python import representative_dataset as repr_dataset
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.platform import test
from tensorflow.python.types import core

def _contains_tensor(sample: repr_dataset.RepresentativeSample) -> bool:
    if False:
        return 10
    'Determines whether `sample` contains any tf.Tensors.\n\n  Args:\n    sample: A `RepresentativeSample`.\n\n  Returns:\n    True iff `sample` contains at least tf.Tensors.\n  '
    return any(map(lambda value: isinstance(value, core.Tensor), sample.values()))

class RepresentativeDatasetTest(test.TestCase):
    """Tests functions for representative datasets."""

    def _assert_tensorlike_all_close(self, sess: session.Session, tensorlike_value_1: core.TensorLike, tensorlike_value_2: core.TensorLike) -> None:
        if False:
            while True:
                i = 10
        'Asserts that two different TensorLike values are "all close".\n\n    Args:\n      sess: Session instance used to evaluate any tf.Tensors.\n      tensorlike_value_1: A TensorLike value.\n      tensorlike_value_2: A TensorLike value.\n    '
        if isinstance(tensorlike_value_1, core.Tensor):
            tensorlike_value_1 = tensorlike_value_1.eval(session=sess)
        if isinstance(tensorlike_value_2, core.Tensor):
            tensorlike_value_2 = tensorlike_value_2.eval(session=sess)
        self.assertAllClose(tensorlike_value_1, tensorlike_value_2)

    def _assert_sample_values_all_close(self, sess: session.Session, repr_ds_1: repr_dataset.RepresentativeDataset, repr_ds_2: repr_dataset.RepresentativeDataset) -> None:
        if False:
            return 10
        'Asserts that the sample values are "all close" between the two datasets.\n\n    This assumes that the order of corresponding samples is preserved and the\n    size of the two datasets are equal.\n\n    Args:\n      sess: Session instance used to evaluate any tf.Tensors.\n      repr_ds_1: A RepresentativeDataset.\n      repr_ds_2: A RepresentativeDataset.\n    '
        for (sample_1, sample_2) in zip(repr_ds_1, repr_ds_2):
            self.assertCountEqual(sample_1.keys(), sample_2.keys())
            for input_key in sample_1:
                self._assert_tensorlike_all_close(sess, sample_1[input_key], sample_2[input_key])

    @test_util.deprecated_graph_mode_only
    def test_replace_tensors_by_numpy_ndarrays_with_tensor_list(self):
        if False:
            while True:
                i = 10
        num_samples = 8
        samples = [np.random.uniform(low=-1.0, high=1.0, size=(3, 3)).astype('f4') for _ in range(num_samples)]
        repr_ds: repr_dataset.RepresentativeDataset = [{'input_tensor': ops.convert_to_tensor(sample)} for sample in samples]
        with self.session() as sess:
            new_repr_ds = repr_dataset.replace_tensors_by_numpy_ndarrays(repr_ds, sess)
            self.assertFalse(any(map(_contains_tensor, new_repr_ds)))
            self._assert_sample_values_all_close(sess, repr_ds, new_repr_ds)

    @test_util.deprecated_graph_mode_only
    def test_replace_tensors_by_numpy_ndarrays_with_tensor_generator(self):
        if False:
            i = 10
            return i + 15
        num_samples = 8
        samples = [np.random.uniform(low=-1.0, high=1.0, size=(1, 4)).astype('f4') for _ in range(num_samples)]

        def data_gen() -> repr_dataset.RepresentativeDataset:
            if False:
                for i in range(10):
                    print('nop')
            for sample in samples:
                yield {'input_tensor': ops.convert_to_tensor(sample)}
        with self.session() as sess:
            new_repr_ds = repr_dataset.replace_tensors_by_numpy_ndarrays(data_gen(), sess)
            self.assertFalse(any(map(_contains_tensor, new_repr_ds)))
            self._assert_sample_values_all_close(sess, data_gen(), new_repr_ds)

    @test_util.deprecated_graph_mode_only
    def test_replace_tensors_by_numpy_ndarrays_is_noop_when_no_tensor(self):
        if False:
            return 10
        repr_ds: repr_dataset.RepresentativeDataset = [{'input_tensor': np.random.uniform(low=-1.0, high=1.0, size=(4, 3))} for _ in range(8)]
        with self.session() as sess:
            new_repr_ds = repr_dataset.replace_tensors_by_numpy_ndarrays(repr_ds, sess)
            self.assertFalse(any(map(_contains_tensor, new_repr_ds)))
            self._assert_sample_values_all_close(sess, repr_ds, new_repr_ds)

    @test_util.deprecated_graph_mode_only
    def test_replace_tensors_by_numpy_ndarrays_mixed_tensor_and_ndarray(self):
        if False:
            return 10
        num_tensors = 4
        samples = [np.random.uniform(low=-1.0, high=1.0, size=(3, 3)).astype('f4') for _ in range(num_tensors)]
        repr_ds: repr_dataset.RepresentativeDataset = [{'tensor_key': ops.convert_to_tensor(sample)} for sample in samples]
        repr_ds.extend([{'tensor_key': np.random.uniform(low=-1.0, high=1.0, size=(3, 3))} for _ in range(4)])
        random.shuffle(repr_ds)
        with self.session() as sess:
            new_repr_ds = repr_dataset.replace_tensors_by_numpy_ndarrays(repr_ds, sess)
            self.assertFalse(any(map(_contains_tensor, new_repr_ds)))
            self._assert_sample_values_all_close(sess, repr_ds, new_repr_ds)

    def test_get_num_samples_returns_num_samples_when_list(self):
        if False:
            while True:
                i = 10
        num_samples = 8
        repr_ds = [{'input': np.random.uniform(low=-1.0, high=1.0, size=(1, 2))} for _ in range(num_samples)]
        self.assertEqual(repr_dataset.get_num_samples(repr_ds), num_samples)

    def test_get_num_samples_returns_none_for_generator(self):
        if False:
            return 10
        num_samples = 8

        def data_gen() -> repr_dataset.RepresentativeDataset:
            if False:
                print('Hello World!')
            for _ in range(num_samples):
                yield {'input_tensor': np.random.uniform(low=-1.0, high=1.0, size=(1, 4))}
        repr_ds = data_gen()
        self.assertIsNone(repr_dataset.get_num_samples(repr_ds))
        self.assertLen(list(repr_ds), num_samples)

    def test_get_num_samples_returns_none_when_len_raises_error(self):
        if False:
            print('Hello World!')

        class LenRaisingError:
            """A test-only class that raises an error when len() is called.

      This mocks the behavior of an Iterator whose size cannot be determined.
      One example is `tf.data.Dataset` whose samples are generated by a
      Generator.
      """

            def __len__(self):
                if False:
                    return 10
                raise ValueError('You cannot take the len() of instance of LenRaisingError.')
        self.assertIsNone(repr_dataset.get_num_samples(LenRaisingError()))

    @test_util.deprecated_graph_mode_only
    def test_create_feed_dict_from_input_data(self):
        if False:
            i = 10
            return i + 15
        signature_def = meta_graph_pb2.SignatureDef(inputs={'input_tensor': meta_graph_pb2.TensorInfo(name='input:0')})
        rng = np.random.default_rng(seed=14)
        input_tensor_value = rng.random(size=(2, 2))
        sample = {'input_tensor': input_tensor_value}
        feed_dict = repr_dataset.create_feed_dict_from_input_data(sample, signature_def)
        self.assertLen(feed_dict, 1)
        self.assertIn('input:0', feed_dict)
        self.assertAllEqual(feed_dict['input:0'], input_tensor_value)

    @test_util.deprecated_graph_mode_only
    def test_create_feed_dict_from_input_data_core_tensors(self):
        if False:
            for i in range(10):
                print('nop')
        signature_def = meta_graph_pb2.SignatureDef(inputs={'input_tensor': meta_graph_pb2.TensorInfo(name='input:0')})
        with self.session():
            input_tensor = constant_op.constant([1, 2, 3, 4, 5, 6])
            sample = {'input_tensor': input_tensor}
            feed_dict = repr_dataset.create_feed_dict_from_input_data(sample, signature_def)
            input_tensor_data = input_tensor.eval()
        self.assertLen(feed_dict, 1)
        self.assertIn('input:0', feed_dict)
        self.assertIsInstance(feed_dict['input:0'], np.ndarray)
        self.assertAllEqual(feed_dict['input:0'], input_tensor_data)

    @test_util.deprecated_graph_mode_only
    def test_create_feed_dict_from_input_data_empty(self):
        if False:
            for i in range(10):
                print('nop')
        signature_def = meta_graph_pb2.SignatureDef(inputs={'input_tensor': meta_graph_pb2.TensorInfo(name='input:0')})
        sample = {}
        feed_dict = repr_dataset.create_feed_dict_from_input_data(sample, signature_def)
        self.assertEmpty(feed_dict)

class RepresentativeDatasetSaverTest(test.TestCase):
    """Test cases for RepresentativeDatasetSaver."""

    def test_save_raises_error(self):
        if False:
            i = 10
            return i + 15
        saver = repr_dataset.RepresentativeDatasetSaver()
        repr_ds = {'serving_default': []}
        with self.assertRaisesRegex(NotImplementedError, 'Method "save" is not implemented.'):
            saver.save(repr_ds)

class TfRecordRepresentativeDatasetTest(test.TestCase):
    """Test cases for RepresentativeDatasetLoader."""

    def test_tf_record_saver_with_generator_dataset(self):
        if False:
            i = 10
            return i + 15
        tf_record_path = self.create_tempfile().full_path
        path_map = {'serving_default': tf_record_path}
        num_samples = 2

        def data_gen():
            if False:
                i = 10
                return i + 15
            for _ in range(num_samples):
                yield {'x': [1, 2]}
        repr_ds_map = {'serving_default': data_gen()}
        saver = repr_dataset.TfRecordRepresentativeDatasetSaver(path_map)
        dataset_file_map = saver.save(repr_ds_map)
        self.assertCountEqual(dataset_file_map.keys(), ['serving_default'])
        dataset_map = repr_dataset.RepresentativeDatasetLoader(dataset_file_map).load()
        self.assertCountEqual(dataset_map.keys(), ['serving_default'])
        samples = dataset_map['serving_default']
        for sample in samples:
            self.assertCountEqual(sample.keys(), {'x'})
            self.assertAllEqual(sample['x'], np.array([1, 2]))
        self.assertLen(samples, num_samples)

    def test_tf_record_saver_when_signature_def_key_mismatch_raises_error(self):
        if False:
            for i in range(10):
                print('nop')
        tf_record_path = self.create_tempfile().full_path
        representative_dataset = [{'x': [2]}]
        repr_ds_map = {'my_signature_key': representative_dataset}
        path_map = {'different_signature_key': tf_record_path}
        saver = repr_dataset.TfRecordRepresentativeDatasetSaver(path_map)
        with self.assertRaisesRegex(ValueError, 'SignatureDef key does not exist in the provided path_map: my_signature_key'):
            saver.save(repr_ds_map)
if __name__ == '__main__':
    test.main()