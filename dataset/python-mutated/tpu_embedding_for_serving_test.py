"""Tests for TPU Embeddings mid level API on CPU."""
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_for_serving
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.util import nest

class TPUEmbeddingForServingTest(test.TestCase):

    def setUp(self):
        if False:
            print('Hello World!')
        super(TPUEmbeddingForServingTest, self).setUp()
        self.embedding_values = np.array(list(range(32)), dtype=np.float64)
        self.initializer = init_ops_v2.Constant(self.embedding_values)
        self.table_video = tpu_embedding_v2_utils.TableConfig(vocabulary_size=8, dim=4, initializer=self.initializer, combiner='sum', name='video')
        self.table_user = tpu_embedding_v2_utils.TableConfig(vocabulary_size=16, dim=2, initializer=self.initializer, combiner='mean', name='user')
        self.feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_video, name='watched'), tpu_embedding_v2_utils.FeatureConfig(table=self.table_video, name='favorited'), tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends'))
        self.batch_size = 2
        self.data_batch_size = 4
        self.feature_watched_indices = [[0, 0], [1, 0], [1, 1], [2, 0], [2, 1], [3, 0]]
        self.feature_watched_values = [0, 0, 1, 0, 1, 1]
        self.feature_watched_row_lengths = [1, 2, 2, 1]
        self.feature_favorited_indices = [[0, 0], [0, 1], [1, 0], [2, 0], [3, 0], [3, 1]]
        self.feature_favorited_values = [0, 1, 1, 0, 0, 1]
        self.feature_favorited_row_lengths = [2, 1, 1, 2]
        self.feature_friends_indices = [[0, 0], [1, 0], [1, 1], [1, 2], [2, 0], [3, 0], [3, 1], [3, 2]]
        self.feature_friends_values = [3, 0, 1, 2, 3, 0, 1, 2]
        self.feature_friends_row_lengths = [1, 3, 1, 3]

    def _create_mid_level(self):
        if False:
            while True:
                i = 10
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        return tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=self.feature_config, optimizer=optimizer)

    def _get_dense_tensors(self, dtype=dtypes.int32):
        if False:
            for i in range(10):
                print('nop')
        feature0 = constant_op.constant(self.feature_watched_values, dtype=dtype)
        feature1 = constant_op.constant(self.feature_favorited_values, dtype=dtype)
        feature2 = constant_op.constant(self.feature_friends_values, dtype=dtype)
        return (feature0, feature1, feature2)

    def test_cpu_dense_lookup(self):
        if False:
            for i in range(10):
                print('nop')
        mid_level = self._create_mid_level()
        features = self._get_dense_tensors()
        results = mid_level(features, weights=None)
        all_lookups = []
        for (feature, config) in zip(nest.flatten(features), self.feature_config):
            table = mid_level.embedding_tables[config.table].numpy()
            all_lookups.append(table[feature.numpy()])
        self.assertAllClose(results, nest.pack_sequence_as(results, all_lookups))

    def test_cpu_dense_lookup_with_weights(self):
        if False:
            return 10
        mid_level = self._create_mid_level()
        features = self._get_dense_tensors()
        weights = self._get_dense_tensors(dtype=dtypes.float32)
        with self.assertRaisesRegex(ValueError, 'Weight specified for .*, but input is dense.'):
            mid_level(features, weights=weights)

    def _get_sparse_tensors(self, dtype=dtypes.int32):
        if False:
            return 10
        feature0 = sparse_tensor.SparseTensor(indices=self.feature_watched_indices, values=constant_op.constant(self.feature_watched_values, dtype=dtype), dense_shape=[self.data_batch_size, 2])
        feature1 = sparse_tensor.SparseTensor(indices=self.feature_favorited_indices, values=constant_op.constant(self.feature_favorited_values, dtype=dtype), dense_shape=[self.data_batch_size, 2])
        feature2 = sparse_tensor.SparseTensor(indices=self.feature_friends_indices, values=constant_op.constant(self.feature_friends_values, dtype=dtype), dense_shape=[self.data_batch_size, 3])
        return (feature0, feature1, feature2)

    def test_cpu_sparse_lookup(self):
        if False:
            for i in range(10):
                print('nop')
        mid_level = self._create_mid_level()
        features = self._get_sparse_tensors()
        results = mid_level(features, weights=None)
        reduced = []
        for (feature, config) in zip(nest.flatten(features), self.feature_config):
            table = mid_level.embedding_tables[config.table].numpy()
            all_lookups = table[feature.values.numpy()]
            ragged = ragged_tensor.RaggedTensor.from_sparse(feature)
            row_starts = ragged.row_starts().numpy()
            reduced.append(np.add.reduceat(all_lookups, row_starts))
            if config.table.combiner == 'mean':
                reduced[-1] /= np.expand_dims(ragged.row_lengths().numpy(), axis=1)
        self.assertAllClose(results, nest.pack_sequence_as(results, reduced))

    def test_cpu_sparse_lookup_with_weights(self):
        if False:
            print('Hello World!')
        mid_level = self._create_mid_level()
        features = self._get_sparse_tensors()
        weights = self._get_sparse_tensors(dtype=dtypes.float32)
        results = mid_level(features, weights=weights)
        weighted_sum = []
        for (feature, weight, config) in zip(nest.flatten(features), nest.flatten(weights), self.feature_config):
            table = mid_level.embedding_tables[config.table].numpy()
            weight = np.expand_dims(weight.values.numpy(), axis=1)
            all_lookups = table[feature.values.numpy()] * weight
            row_starts = ragged_tensor.RaggedTensor.from_sparse(feature).row_starts()
            row_starts = row_starts.numpy()
            weighted_sum.append(np.add.reduceat(all_lookups, row_starts))
            if config.table.combiner == 'mean':
                weighted_sum[-1] /= np.add.reduceat(weight, row_starts)
        self.assertAllClose(results, nest.pack_sequence_as(results, weighted_sum))

    def test_cpu_sparse_lookup_with_non_sparse_weights(self):
        if False:
            for i in range(10):
                print('nop')
        mid_level = self._create_mid_level()
        features = self._get_sparse_tensors()
        weights = self._get_dense_tensors(dtype=dtypes.float32)
        with self.assertRaisesRegex(ValueError, 'but it does not match type of the input which is'):
            mid_level(features, weights=weights)

    def _get_ragged_tensors(self, dtype=dtypes.int32):
        if False:
            print('Hello World!')
        feature0 = ragged_tensor.RaggedTensor.from_row_lengths(values=constant_op.constant(self.feature_watched_values, dtype=dtype), row_lengths=self.feature_watched_row_lengths)
        feature1 = ragged_tensor.RaggedTensor.from_row_lengths(values=constant_op.constant(self.feature_favorited_values, dtype=dtype), row_lengths=self.feature_favorited_row_lengths)
        feature2 = ragged_tensor.RaggedTensor.from_row_lengths(values=constant_op.constant(self.feature_friends_values, dtype=dtype), row_lengths=self.feature_friends_row_lengths)
        return (feature0, feature1, feature2)

    def test_cpu_ragged_lookup_with_weights(self):
        if False:
            return 10
        mid_level = self._create_mid_level()
        features = self._get_ragged_tensors()
        weights = self._get_ragged_tensors(dtype=dtypes.float32)
        results = mid_level(features, weights=weights)
        weighted_sum = []
        for (feature, weight, config) in zip(nest.flatten(features), nest.flatten(weights), self.feature_config):
            table = mid_level.embedding_tables[config.table].numpy()
            weight = np.expand_dims(weight.values.numpy(), axis=1)
            all_lookups = table[feature.values.numpy()] * weight
            row_starts = feature.row_starts().numpy()
            weighted_sum.append(np.add.reduceat(all_lookups, row_starts))
            if config.table.combiner == 'mean':
                weighted_sum[-1] /= np.add.reduceat(weight, row_starts)
        self.assertAllClose(results, nest.pack_sequence_as(results, weighted_sum))

    def test_cpu_invalid_structure_for_features(self):
        if False:
            while True:
                i = 10
        mid_level = self._create_mid_level()
        features = tuple(self._get_sparse_tensors()[:2])
        with self.assertRaises(ValueError):
            mid_level(features, weights=None)

    def test_cpu_invalid_structure_for_weights(self):
        if False:
            return 10
        mid_level = self._create_mid_level()
        features = self._get_sparse_tensors()
        weights = tuple(self._get_dense_tensors(dtype=dtypes.float32)[:2])
        with self.assertRaises(ValueError):
            mid_level(features, weights=weights)

    def _numpy_sequence_lookup(self, table, indices, values, batch_size, max_sequence_length, dim):
        if False:
            print('Hello World!')
        valid_entries = np.nonzero(indices[:, 1] < max_sequence_length)[0]
        indices = indices[valid_entries]
        values = values[valid_entries]
        lookup = table[values]
        scatter_result = np.zeros([batch_size, max_sequence_length, dim])
        for (i, index) in enumerate(indices):
            scatter_result[index[0], index[1], :] = lookup[i]
        return scatter_result

    def test_cpu_sequence_lookup_sparse(self):
        if False:
            print('Hello World!')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends', max_sequence_length=2),)
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=optimizer)
        features = self._get_sparse_tensors()[2:3]
        result = mid_level(features, weights=None)
        golden = self._numpy_sequence_lookup(mid_level.embedding_tables[self.table_user].numpy(), features[0].indices.numpy(), features[0].values.numpy(), self.data_batch_size, feature_config[0].max_sequence_length, self.table_user.dim)
        self.assertAllClose(result[0], golden)

    def test_cpu_sequence_lookup_ragged(self):
        if False:
            print('Hello World!')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends', max_sequence_length=2),)
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=optimizer)
        features = self._get_ragged_tensors()[2:3]
        result = mid_level(features, weights=None)
        sparse_ver = features[0].to_sparse()
        golden = self._numpy_sequence_lookup(mid_level.embedding_tables[self.table_user].numpy(), sparse_ver.indices.numpy(), sparse_ver.values.numpy(), self.data_batch_size, feature_config[0].max_sequence_length, self.table_user.dim)
        self.assertAllClose(result[0], golden)

    def test_cpu_high_dimensional_lookup_ragged(self):
        if False:
            for i in range(10):
                print('nop')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends', output_shape=[2, 2]),)
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=optimizer)
        features = self._get_ragged_tensors()[2:3]
        result = mid_level(features, weights=None)
        self.assertAllClose(result[0].shape, (2, 2, 2))

    def test_cpu_high_dimensional_sequence_lookup_ragged(self):
        if False:
            for i in range(10):
                print('nop')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends', output_shape=[2, 4]),)
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=optimizer)
        features = self._get_ragged_tensors()[2:3]
        result = mid_level(features, weights=None)
        self.assertAllClose(result[0].shape, (2, 4, 2))

    def test_cpu_high_dimensional_invalid_lookup_ragged(self):
        if False:
            while True:
                i = 10
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends', output_shape=[3]),)
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=optimizer)
        features = self._get_ragged_tensors()[2:3]
        with self.assertRaisesRegex(ValueError, 'Output shape set in the FeatureConfig should be the factor'):
            mid_level(features, weights=None)

    def test_cpu_no_optimizer(self):
        if False:
            for i in range(10):
                print('nop')
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_video, name='watched', max_sequence_length=2),)
        mid_level = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=None)
        mid_level.build()
        self.assertEqual(list(mid_level._variables[self.table_video.name].keys()), ['parameters'])

    def test_cpu_multiple_creation(self):
        if False:
            return 10
        feature_config = (tpu_embedding_v2_utils.FeatureConfig(table=self.table_user, name='friends', max_sequence_length=2),)
        optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        embedding_one = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=optimizer)
        embedding_two = tpu_embedding_for_serving.TPUEmbeddingForServing(feature_config=feature_config, optimizer=optimizer)
        embedding_one.build()
        embedding_two.build()
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()