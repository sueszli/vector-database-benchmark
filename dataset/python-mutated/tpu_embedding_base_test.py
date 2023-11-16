"""Base Class for TPU Embedding tests."""
import os
from typing import Tuple
from absl import flags
from absl.testing import parameterized
import numpy as np
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.distribute.cluster_resolver import tpu_cluster_resolver
from tensorflow.python.eager import remote
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.platform import test
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.util import nest
FLAGS = flags.FLAGS
flags.DEFINE_string('tpu', '', 'Name of TPU to connect to.')
flags.DEFINE_string('project', None, 'Name of GCP project with TPU.')
flags.DEFINE_string('zone', None, 'Name of GCP zone with TPU.')
flags.DEFINE_string('model_dir', os.environ.get('TEST_TMPDIR'), 'A temporary directory.')

class TPUEmbeddingBaseTest(parameterized.TestCase, test.TestCase):

    def skip_if_oss(self):
        if False:
            return 10
        if FLAGS.project is not None or FLAGS.zone is not None:
            self.skipTest('Skipping tests for oss as it is slow to run every test in cloud tpu.')

    def setUp(self):
        if False:
            while True:
                i = 10
        super(TPUEmbeddingBaseTest, self).setUp()
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
        self.resolver = None

        def create_hight_dimensional_indices(indices):
            if False:
                while True:
                    i = 10
            indices = np.array(indices, dtype=np.int32)
            batch_size_index = np.repeat(np.arange(self.data_batch_size), len(indices)).reshape(-1, 1)
            repeated_indices = np.tile(indices, (self.data_batch_size, 1))
            return np.concatenate([batch_size_index, repeated_indices], axis=1)
        self.feature_watched_indices_high_dimensional = create_hight_dimensional_indices(self.feature_watched_indices)
        self.feature_watched_values_high_dimensional = self.feature_watched_values * self.data_batch_size
        self.feature_watched_row_lengths_high_dimensional = self.feature_watched_row_lengths * self.data_batch_size
        self.feature_favorited_indices_high_dimensional = create_hight_dimensional_indices(self.feature_favorited_indices)
        self.feature_favorited_values_high_dimensional = self.feature_favorited_values * self.data_batch_size
        self.feature_favorited_row_lengths_high_dimensional = self.feature_favorited_row_lengths * self.data_batch_size
        self.feature_friends_indices_high_dimensional = create_hight_dimensional_indices(self.feature_friends_indices)
        self.feature_friends_values_high_dimensional = self.feature_friends_values * self.data_batch_size
        self.feature_friends_row_lengths_high_dimensional = self.feature_friends_row_lengths * self.data_batch_size

    def _init_tpu_system(self):
        if False:
            return 10
        self.resolver = tpu_cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu, zone=FLAGS.zone, project=FLAGS.project)
        if hasattr(self.resolver, '_cloud_tpu_client'):
            self.resolver._cloud_tpu_client.configure_tpu_version(version='nightly', restart_type='always')
        remote.connect_to_cluster(self.resolver)
        return tpu_cluster_resolver.initialize_tpu_system(self.resolver)

    def _get_strategy(self):
        if False:
            while True:
                i = 10
        _ = self._init_tpu_system()
        return tpu_strategy.TPUStrategy(self.resolver)

    def _create_mid_level(self, optimizer=None):
        if False:
            print('Hello World!')
        if optimizer is None:
            optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
        return tpu_embedding_v2.TPUEmbedding(feature_config=self.feature_config, optimizer=optimizer)

    def _create_strategy_and_mid_level(self, optimizer_name) -> Tuple[tpu_strategy.TPUStrategy, tpu_embedding_v2.TPUEmbedding, tpu_embedding_v2_utils._Optimizer]:
        if False:
            while True:
                i = 10
        strategy = self._get_strategy()
        with strategy.scope():
            if optimizer_name == 'sgd':
                optimizer = tpu_embedding_v2_utils.SGD(learning_rate=0.1)
            elif optimizer_name == 'adagrad':
                optimizer = tpu_embedding_v2_utils.Adagrad(learning_rate=0.1)
            elif optimizer_name == 'adam':
                optimizer = tpu_embedding_v2_utils.Adam(learning_rate=0.1)
            elif optimizer_name == 'ftrl':
                optimizer = tpu_embedding_v2_utils.FTRL(learning_rate=0.1)
            elif optimizer_name == 'adagrad_momentum':
                optimizer = tpu_embedding_v2_utils.AdagradMomentum(learning_rate=0.1, momentum=0.9, use_nesterov=True, exponent=3.0, epsilon=0.1, beta2=0.9)
            else:
                raise ValueError('optimizer is not recognized: ', optimizer_name)
            mid_level_api = self._create_mid_level(optimizer=optimizer)
        return (strategy, mid_level_api, optimizer)

    def _create_sparse_data(self, include_weights, weight=0.5):
        if False:
            for i in range(10):
                print('nop')
        sparse_features = (sparse_tensor.SparseTensor(indices=self.feature_watched_indices, values=self.feature_watched_values, dense_shape=[self.data_batch_size, 2]), sparse_tensor.SparseTensor(indices=self.feature_favorited_indices, values=self.feature_favorited_values, dense_shape=[self.data_batch_size, 2]), sparse_tensor.SparseTensor(indices=self.feature_friends_indices, values=self.feature_friends_values, dense_shape=[self.data_batch_size, 3]))
        if include_weights:
            weights = []
            for sparse in sparse_features:
                values = array_ops.ones_like(sparse.values, dtype=dtypes.float32) * weight
                weights.append(sparse_tensor.SparseTensor(indices=sparse.indices, values=values, dense_shape=sparse.dense_shape))
            sparse_features = (sparse_features, tuple(weights))
        return sparse_features

    def _create_sparse_dataset(self, strategy, include_weights=False, weight=0.5):
        if False:
            print('Hello World!')
        sparse_features = self._create_sparse_data(include_weights, weight)
        dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
        return dataset.unbatch().repeat().batch(self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

    def _create_high_dimensional_sparse_dataset(self, strategy, include_weights=False, weight=0.5):
        if False:
            print('Hello World!')
        sparse_features = (sparse_tensor.SparseTensor(indices=self.feature_watched_indices_high_dimensional, values=self.feature_watched_values_high_dimensional, dense_shape=[self.data_batch_size, self.data_batch_size, 2]), sparse_tensor.SparseTensor(indices=self.feature_favorited_indices_high_dimensional, values=self.feature_favorited_values_high_dimensional, dense_shape=[self.data_batch_size, self.data_batch_size, 2]), sparse_tensor.SparseTensor(indices=self.feature_friends_indices_high_dimensional, values=self.feature_friends_values_high_dimensional, dense_shape=[self.data_batch_size, self.data_batch_size, 3]))
        if include_weights:
            weights = []
            for sparse in sparse_features:
                values = array_ops.ones_like(sparse.values, dtype=dtypes.float32) * weight
                weights.append(sparse_tensor.SparseTensor(indices=sparse.indices, values=values, dense_shape=sparse.dense_shape))
            sparse_features = (sparse_features, tuple(weights))
        dataset = dataset_ops.DatasetV2.from_tensors(sparse_features)
        return dataset.unbatch().repeat().batch(self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

    def _create_high_dimensional_ragged_dataset(self, strategy, include_weights=False, weight=0.5):
        if False:
            return 10
        ragged_features = (ragged_tensor.RaggedTensor.from_row_lengths(row_lengths=self.feature_watched_row_lengths_high_dimensional, values=self.feature_watched_values_high_dimensional), ragged_tensor.RaggedTensor.from_row_lengths(row_lengths=self.feature_favorited_row_lengths_high_dimensional, values=self.feature_favorited_values_high_dimensional), ragged_tensor.RaggedTensor.from_row_lengths(row_lengths=self.feature_friends_row_lengths_high_dimensional, values=self.feature_friends_values_high_dimensional))
        if include_weights:
            weights = []
            for ragged in ragged_features:
                values = array_ops.ones_like(ragged.values, dtype=dtypes.float32) * weight
                weights.append(ragged_tensor.RaggedTensor(row_lengths=ragged.row_lengths(), values=values))
            ragged_features = (ragged_features, tuple(weights))
        dataset = dataset_ops.DatasetV2.from_tensors(ragged_features)
        return dataset.unbatch().repeat().batch(self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

    def _create_ragged_dataset(self, strategy, include_weights=False, weight=0.5):
        if False:
            while True:
                i = 10
        sparse_features = self._create_sparse_data(include_weights, weight)
        ragged_features = nest.map_structure(ragged_tensor.RaggedTensor.from_sparse, sparse_features)
        dataset = dataset_ops.DatasetV2.from_tensors(ragged_features)
        return dataset.unbatch().repeat().batch(self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

    def _create_dense_dataset(self, strategy, include_weights=False, weight=0.5):
        if False:
            while True:
                i = 10
        features = (constant_op.constant(self.feature_watched_values[:self.data_batch_size], dtype=dtypes.int32), constant_op.constant(self.feature_favorited_values[:self.data_batch_size], dtype=dtypes.int32), constant_op.constant(self.feature_friends_values[:self.data_batch_size], dtype=dtypes.int32))
        if include_weights:
            weights = [array_ops.ones_like(t, dtype=dtypes.float32) * weight for t in features]
            features = (features, tuple(weights))
        dataset = dataset_ops.DatasetV2.from_tensors(features)
        return dataset.unbatch().repeat().batch(self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

    def _create_high_dimensional_dense_dataset(self, strategy, include_weights=False, weight=0.5):
        if False:
            i = 10
            return i + 15
        dense_size = self.data_batch_size * self.data_batch_size
        features = (constant_op.constant(self.feature_watched_values_high_dimensional[:dense_size], shape=(self.data_batch_size, self.data_batch_size, 1), dtype=dtypes.int32), constant_op.constant(self.feature_favorited_values_high_dimensional[:dense_size], shape=(self.data_batch_size, self.data_batch_size, 1), dtype=dtypes.int32), constant_op.constant(self.feature_friends_values_high_dimensional[:dense_size], shape=(self.data_batch_size, self.data_batch_size, 1), dtype=dtypes.int32))
        if include_weights:
            weights = [array_ops.ones_like(t, dtype=dtypes.float32) * weight for t in features]
            features = (features, tuple(weights))
        dataset = dataset_ops.DatasetV2.from_tensors(features)
        return dataset.unbatch().repeat().batch(self.batch_size * strategy.num_replicas_in_sync, drop_remainder=True)

    def _check_results(self, strategy, shard_out_val, training, input_data, table_to_variable, optimizer, is_high_dimensional):
        if False:
            while True:
                i = 10
        num_replicas = strategy.num_replicas_in_sync
        loss = self._unpack(strategy, shard_out_val[0])
        activation_watched = self._unpack(strategy, shard_out_val[1])
        activation_favorited = self._unpack(strategy, shard_out_val[2])
        activation_friends = self._unpack(strategy, shard_out_val[3])
        activation_watched_gold0 = np.array([[0, 1, 2, 3], [4, 6, 8, 10]])
        activation_favorited_gold0 = np.array([[4, 6, 8, 10], [4, 5, 6, 7]])
        activation_friends_gold0 = np.array([[6, 7], [2, 3]])
        loss_gold0 = self._compute_loss(activation_watched_gold0, activation_favorited_gold0, activation_friends_gold0)
        activation_watched_gold = np.concatenate((activation_watched_gold0, activation_favorited_gold0))
        activation_favorited_gold = np.concatenate((activation_favorited_gold0, activation_watched_gold0))
        activation_friends_gold = np.concatenate((activation_friends_gold0, activation_friends_gold0))
        if is_high_dimensional:
            activation_watched_gold = np.stack([activation_watched_gold] * self.batch_size * num_replicas)
            activation_favorited_gold = np.stack([activation_favorited_gold] * self.batch_size * num_replicas)
            activation_friends_gold = np.stack([activation_friends_gold] * self.batch_size * num_replicas)
        elif num_replicas == 1:
            activation_watched_gold = activation_watched_gold0
            activation_favorited_gold = activation_favorited_gold0
            activation_friends_gold = activation_friends_gold0
        else:
            activation_watched_gold = np.concatenate([activation_watched_gold] * (num_replicas // self.batch_size))
            activation_favorited_gold = np.concatenate([activation_favorited_gold] * (num_replicas // self.batch_size))
            activation_friends_gold = np.concatenate([activation_friends_gold] * (num_replicas // self.batch_size))
        loss_gold = [loss_gold0] * num_replicas
        self.assertAllClose(activation_watched_gold, activation_watched)
        self.assertAllClose(activation_favorited_gold, activation_favorited)
        self.assertAllClose(activation_friends_gold, activation_friends)
        self.assertAllClose(loss_gold, loss)
        embedding_table_video_before = np.copy(np.reshape(self.embedding_values, [8, 4]))
        embedding_table_user_before = np.copy(np.reshape(self.embedding_values, [16, 2]))
        if is_high_dimensional:
            global_batch_size = self.batch_size * self.data_batch_size * num_replicas
        else:
            global_batch_size = self.batch_size * num_replicas
        if training:
            gradient_wrt_watched_gold = 2 * activation_watched_gold / global_batch_size
            gradient_wrt_favorited_gold = 2 * activation_favorited_gold / global_batch_size
            gradient_wrt_friends_gold = 2 * activation_friends_gold / global_batch_size
            gradients_wrt_user = self._compute_gradients_wrt_embedding_table(gradient_wrt_friends_gold, embedding_table_user_before, input_data[2].indices.numpy(), input_data[2].values.numpy(), self.table_user.combiner)
            gradients_wrt_video = self._compute_gradients_wrt_embedding_table(gradient_wrt_favorited_gold, embedding_table_video_before, input_data[1].indices.numpy(), input_data[1].values.numpy(), self.table_video.combiner) + self._compute_gradients_wrt_embedding_table(gradient_wrt_watched_gold, embedding_table_video_before, input_data[0].indices.numpy(), input_data[0].values.numpy(), self.table_video.combiner)
            self._check_embedding_and_slot_variables(embedding_table_user_before, gradients_wrt_user, embedding_table_video_before, gradients_wrt_video, optimizer, table_to_variable)

    def _check_embedding_and_slot_variables(self, embedding_table_user_before, gradients_wrt_user, embedding_table_video_before, gradients_wrt_video, optimizer, table_to_variable):
        if False:
            return 10
        if isinstance(optimizer, tpu_embedding_v2_utils.SGD):
            check_fn = self._check_embedding_and_slot_variables_for_sgd
        elif isinstance(optimizer, tpu_embedding_v2_utils.Adagrad):
            check_fn = self._check_embedding_and_slot_variables_for_adagrad
        elif isinstance(optimizer, tpu_embedding_v2_utils.AdagradMomentum):
            check_fn = self._check_embedding_and_slot_variables_for_adagrad_momentum
        elif isinstance(optimizer, tpu_embedding_v2_utils.Adam):
            check_fn = self._check_embedding_and_slot_variables_for_adam
        elif isinstance(optimizer, tpu_embedding_v2_utils.FTRL):
            check_fn = self._check_embedding_and_slot_variables_for_ftrl
        else:
            raise ValueError('optimizer is not recognized: ', type(optimizer))
        check_fn(embedding_table_user_before, gradients_wrt_user, optimizer, table_to_variable[self.table_user.name])
        check_fn(embedding_table_video_before, gradients_wrt_video, optimizer, table_to_variable[self.table_video.name])

    def _check_embedding_and_slot_variables_for_sgd(self, embedding_table_before, gradients, optimizer, variables):
        if False:
            print('Hello World!')
        embedding_table = np.copy(embedding_table_before)
        embedding_table -= optimizer.learning_rate * np.sum(gradients, axis=0)
        self.assertAllClose(self._get_variable(variables['parameters']).numpy(), embedding_table)

    def _check_embedding_and_slot_variables_for_adagrad(self, embedding_table_before, gradients, optimizer, variable):
        if False:
            print('Hello World!')
        embedding_table = np.copy(embedding_table_before)
        accumulator = optimizer.initial_accumulator_value + np.sum(gradients, axis=0) ** 2
        embedding_table -= optimizer.learning_rate * np.sum(gradients, axis=0) / np.sqrt(accumulator)
        self.assertAllClose(self._get_variable(variable['parameters']).numpy(), embedding_table)
        self.assertAllClose(self._get_variable(variable['accumulators']).numpy(), accumulator)

    def _check_embedding_and_slot_variables_for_adagrad_momentum(self, embedding_table_before, gradients, optimizer, variable):
        if False:
            return 10
        embedding_table = np.copy(embedding_table_before)
        accumulator = np.zeros(self._get_variable(variable['accumulators']).shape)
        momenta = np.zeros(self._get_variable(variable['momenta']).shape)
        gradients = np.sum(gradients, axis=0)
        if optimizer.beta2 == 1.0:
            accumulator += gradients ** 2
        else:
            accumulator = optimizer.beta2 * accumulator + (1 - optimizer.beta2) * gradients ** 2
        accumulator_power = np.power(accumulator + optimizer.epsilon, -1.0 / optimizer.exponent)
        momenta = optimizer.momentum * momenta + gradients * accumulator_power
        if optimizer.use_nesterov:
            update = optimizer.momentum * momenta + gradients * accumulator_power
        else:
            update = momenta
        embedding_table -= optimizer.learning_rate * update
        self.assertAllClose(self._get_variable(variable['parameters']).numpy(), embedding_table, rtol=0.001)
        self.assertAllClose(self._get_variable(variable['accumulators']).numpy(), accumulator, rtol=0.001)
        self.assertAllClose(self._get_variable(variable['momenta']).numpy(), momenta, rtol=0.001)

    def _check_embedding_and_slot_variables_for_adam(self, embedding_table_before, gradients, optimizer, variable):
        if False:
            return 10
        embedding_table = np.copy(embedding_table_before)
        g = np.sum(gradients, axis=0)
        v = g ** 2 * (1 - optimizer.beta_2)
        m = g * (1 - optimizer.beta_1)
        epsilon = optimizer.epsilon
        lr_modifier = 1
        embedding_table -= m * optimizer.learning_rate * lr_modifier / (np.sqrt(v) + epsilon)
        self.assertAllClose(self._get_variable(variable['parameters']).numpy(), embedding_table, rtol=0.0001)
        self.assertAllClose(self._get_variable(variable['momenta']).numpy(), m, rtol=0.0001)
        self.assertAllClose(self._get_variable(variable['velocities']).numpy(), v, rtol=0.0001)

    def _check_embedding_and_slot_variables_for_ftrl(self, embedding_table_before, gradients, optimizer, variable):
        if False:
            print('Hello World!')
        embedding_table = np.copy(embedding_table_before)
        neg_lr_p = -optimizer.learning_rate_power
        accumulator = optimizer.initial_accumulator_value + np.sum(gradients, axis=0) ** 2
        sigma = (accumulator ** neg_lr_p - optimizer.initial_accumulator_value ** neg_lr_p) / optimizer.learning_rate
        linear = np.sum(gradients, axis=0) - sigma * embedding_table
        quadratic = accumulator ** neg_lr_p / optimizer.learning_rate
        embedding_table = -linear / quadratic
        actual_parameters = self._get_variable(variable['parameters']).numpy()
        actual_parameters *= linear != 0.0
        self.assertAllClose(actual_parameters, embedding_table, rtol=5e-05)
        self.assertAllClose(self._get_variable(variable['linears']).numpy(), linear, rtol=0.0005)
        self.assertAllClose(self._get_variable(variable['accumulators']).numpy(), accumulator)

    def _get_replica_numpy(self, structured, strategy, replica_id):
        if False:
            while True:
                i = 10

        def select_replica(x):
            if False:
                for i in range(10):
                    print('nop')
            x = strategy.experimental_local_results(x)
            if len(x) == 1:
                return x.numpy()
            return x[replica_id].numpy()
        return nest.map_structure(select_replica, structured)

    def _compute_gradients_wrt_embedding_table(self, gradient_wrt_activation, embedding_table, feature_indices, feature_values, combiner):
        if False:
            return 10
        "Compute gradients wrt embedding_table.\n\n    Args:\n      gradient_wrt_activation: `np.array` with shape `batch_size` by embedding\n        `dimension`.\n      embedding_table: `np.array` with shape `vocabulary_size` by embedding\n        `dimension`.\n      feature_indices: `indices` as used to construct `SparseTensor`.\n      feature_values: `values` as used to construct `SparseTensor`.\n      combiner: `String`, 'mean' or 'sum'.\n\n    Returns:\n      Gradients wrt `embedding_table`, an `np.array`s with shape\n        `batch_size` by `vocabulary_size` by\n        embedding `dimension`.\n\n    Raises:\n      ValueError: if `combiner` is not one of 'mean' or 'sum'.\n    "
        if combiner not in ('mean', 'sum'):
            raise ValueError('`combiner` must be mean or sum; got {}.'.format(combiner))
        grads_shape = gradient_wrt_activation.shape[:-1] + embedding_table.shape
        grads = np.zeros(shape=grads_shape)
        count = np.zeros(shape=grads_shape)
        for (feature_indice, vocabulary_id) in zip(feature_indices, feature_values):
            batch_index = tuple(feature_indice[:-1])
            grads[batch_index][vocabulary_id] += gradient_wrt_activation[batch_index]
            count[batch_index] += 1
        count[count == 0] = 1
        if combiner == 'mean':
            grads = grads / count
        return np.reshape(grads, (-1, *embedding_table.shape))

    def _unpack(self, strategy, per_replica_output):
        if False:
            return 10
        per_replica_output = strategy.experimental_local_results(per_replica_output)
        per_replica_output = array_ops.concat(per_replica_output, axis=0).numpy()
        return per_replica_output

    def _get_total_loss_tensor(self, activations):
        if False:
            print('Hello World!')
        losses = []
        for activation in activations:
            losses.append(math_ops.reduce_mean(math_ops.reduce_sum(gen_math_ops.squared_difference(activation, 0), axis=-1)))
        total_loss = array_ops.expand_dims_v2(sum(losses), 0)
        return total_loss

    def _compute_loss(self, activation_watched, activation_favorited, activation_friends):
        if False:
            return 10
        watched_loss = np.mean(np.sum(activation_watched ** 2, axis=-1))
        favorited_loss = np.mean(np.sum(activation_favorited ** 2, axis=-1))
        friends_loss = np.mean(np.sum(activation_friends ** 2, axis=-1))
        loss = watched_loss + favorited_loss + friends_loss
        return loss

    def _get_variable(self, variable):
        if False:
            return 10
        if isinstance(variable, tpu_embedding_v2.TPUEmbeddingVariable):
            return variable.variables[0]
        return variable

    def _get_tmpdir(self, name, subdir=''):
        if False:
            i = 10
            return i + 15
        segments = [FLAGS.model_dir, name] + ([subdir] if subdir else [])
        return os.path.join(*segments)