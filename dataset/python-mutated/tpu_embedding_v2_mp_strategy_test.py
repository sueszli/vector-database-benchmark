"""Tests for TPU Embeddings mid level API on TPU."""
from absl.testing import parameterized
import numpy as np
from tensorflow.python.compat import v2_compat
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import tpu_strategy
from tensorflow.python.eager import backprop
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import init_ops_v2
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables as tf_variables
from tensorflow.python.platform import test
from tensorflow.python.tpu import device_assignment as device_lib
from tensorflow.python.tpu import tpu_embedding_v2
from tensorflow.python.tpu import tpu_embedding_v2_utils
from tensorflow.python.tpu.tests import tpu_embedding_base_test

class TPUEmbeddingTPUStrategyV2Test(tpu_embedding_base_test.TPUEmbeddingBaseTest):

    def setUp(self):
        if False:
            print('Hello World!')
        super().setUp()
        self._num_replicas = 1
        self._num_cores_per_replica = 2

    def _get_strategy(self) -> tpu_strategy.TPUStrategy:
        if False:
            print('Hello World!')
        topology = self._init_tpu_system()
        d_assign = device_lib.device_assignment(topology, computation_shape=[1, 1, 1, 2], num_replicas=1)
        self.strategy = tpu_strategy.TPUStrategyV2(self.resolver, experimental_device_assignment=d_assign, experimental_spmd_xla_partitioning=True)
        self.embedding_devices = sum((list(replica) for replica in self.strategy.extended._tpu_devices), [])
        return self.strategy

    def enqueue(self, inp, mid_level_api, use_device, training):
        if False:
            return 10
        if use_device:
            for (emb, device) in zip(inp, self.embedding_devices):
                mid_level_api.enqueue(emb, device=device, training=training)
        else:
            mid_level_api.enqueue(inp[0], training=training)

    @parameterized.parameters(False, True)
    def test_spmd_training(self, use_device):
        if False:
            print('Hello World!')
        num_steps = 10
        num_steps_float = float(num_steps)
        starting_lr = 1.0
        ending_lr = 0.5
        strategy = self._get_strategy()
        with strategy.scope():
            step_counter = tf_variables.Variable(0.0, dtypes.float32)

            def lr_function():
                if False:
                    return 10
                return gen_math_ops.maximum(ending_lr, starting_lr + (ending_lr - starting_lr) * step_counter / num_steps_float)
            optimizer = tpu_embedding_v2_utils.SGD(learning_rate=lr_function)
            table_config = tpu_embedding_v2_utils.TableConfig(vocabulary_size=2, dim=4, initializer=init_ops_v2.Constant(np.zeros((2, 4))), combiner='sum', name='table')
            mid_level_api = tpu_embedding_v2.TPUEmbedding(feature_config={'feature': tpu_embedding_v2_utils.FeatureConfig(table=table_config, name='feature')}, optimizer=optimizer)

        def input_fn(ctx):
            if False:
                print('Hello World!')
            del ctx
            feature = {'feature': constant_op.constant([0, 1], shape=(2, 1), dtype=dtypes.int32)}
            return dataset_ops.DatasetV2.from_tensors(feature).repeat()

        def create_datasets():
            if False:
                for i in range(10):
                    print('nop')
            'Creates either a per-replica dataset, or multiple per-devices ones.\n\n      This function explicitly creates per-device datasets because the strategy\n      does not produce a distributed dataset in the model-parallel case; there\n      is only one replica. Without this consideration, the embeddings would be\n      read as [0, 0] instead of the expected [0, 1] since all the devices would\n      receive the same value.\n\n      Returns:\n        A list of one or more dataset(s).\n      '
            if use_device:
                datasets = []
                for i in range(len(self.embedding_devices)):
                    datasets.append(dataset_ops.DatasetV2.from_tensor_slices({'feature': [[[i % self._num_cores_per_replica]]]}).repeat())
                return datasets
            else:
                dataset = strategy.distribute_datasets_from_function(input_fn, options=distribute_lib.InputOptions(experimental_fetch_to_device=False))
                return [dataset]
        datasets = create_datasets()
        iterators = [iter(ds) for ds in datasets]

        @def_function.function(jit_compile=True)
        def test_fn():
            if False:
                for i in range(10):
                    print('nop')

            def step():
                if False:
                    print('Hello World!')
                with backprop.GradientTape() as tape:
                    activations = mid_level_api.dequeue()
                    tape.watch(activations)
                    result = math_ops.reduce_sum(activations['feature'])
                    loss = result / self._num_replicas
                grads = tape.gradient(loss, activations)
                mid_level_api.apply_gradients(grads)
                return activations
            inp = [next(it) for it in iterators]
            self.enqueue(inp, mid_level_api, use_device, training=True)
            return strategy.run(step)
        results = []
        for _ in range(num_steps):
            result = test_fn()
            results.append(self._unpack(strategy, result['feature']))
            step_counter.assign_add(1.0)
        learning_rates = [starting_lr - (starting_lr - ending_lr) / num_steps * j for j in range(num_steps)]
        cumsum = [sum(learning_rates[0:j]) for j in range(num_steps)]
        goldens = [[[-cumsum[i]] * table_config.dim] * self._num_cores_per_replica for i in range(10)]
        self.assertAllClose(results, goldens)
if __name__ == '__main__':
    v2_compat.enable_v2_behavior()
    test.main()