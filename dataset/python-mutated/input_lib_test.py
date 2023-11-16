"""Tests for the input_lib library."""
import collections
from absl.testing import parameterized
import numpy as np
from tensorflow.python import tf2
from tensorflow.python.data.experimental.ops import data_service_ops
from tensorflow.python.data.experimental.service import server_lib
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.ops.options import AutoShardPolicy
from tensorflow.python.distribute import combinations
from tensorflow.python.distribute import device_util
from tensorflow.python.distribute import distribute_lib
from tensorflow.python.distribute import distribute_utils
from tensorflow.python.distribute import input_lib
from tensorflow.python.distribute import input_ops
from tensorflow.python.distribute import input_util
from tensorflow.python.distribute import multi_worker_util
from tensorflow.python.distribute import reduce_util
from tensorflow.python.distribute import strategy_combinations
from tensorflow.python.distribute import test_util
from tensorflow.python.distribute.v1 import input_lib as input_lib_v1
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import test
from tensorflow.python.framework import composite_tensor
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import extension_type
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util as framework_test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variables
from tensorflow.python.ops.ragged import ragged_tensor as ragged_tensor_lib
from tensorflow.python.util import nest

class DistributedIteratorTestBase(test.TestCase):

    def _wrap_iterator(self, input_type, dataset_or_input_fn, input_workers, devices, num_replicas_in_sync, strategy, input_context=None):
        if False:
            for i in range(10):
                print('nop')
        if input_type == 'input_fn':
            self.assertIsNone(input_context, msg='`The input_context` arg is only used to shard dataset in `MultiWorkerMirroredStrategy` when the input type is dataset.')
            input_contexts = []
            for i in range(input_workers.num_workers):
                input_contexts.append(distribute_lib.InputContext(num_input_pipelines=input_workers.num_workers, input_pipeline_id=i, num_replicas_in_sync=len(devices)))
            iterator = input_lib_v1.InputFunctionIterator(dataset_or_input_fn, input_workers, input_contexts, strategy)
        else:
            iterator = input_lib_v1.DatasetIterator(dataset_or_input_fn, input_workers, strategy, num_replicas_in_sync=num_replicas_in_sync, input_context=input_context)
        return iterator

    def _wrap_dataset(self, input_type, dataset, input_workers, num_replicas_in_sync, strategy, input_context=None):
        if False:
            return 10
        if input_type == 'dataset':
            if tf2.enabled():
                return input_lib.DistributedDataset(input_workers, strategy, dataset, num_replicas_in_sync=num_replicas_in_sync, input_context=input_context)
            else:
                return input_lib_v1.DistributedDatasetV1(dataset, input_workers, strategy, num_replicas_in_sync=num_replicas_in_sync, input_context=input_context)
        else:
            return strategy.distribute_datasets_from_function(dataset)

    def _assert_iterator_values(self, iterator, expected_values, evaluate_fn, devices, enable_get_next_as_optional=False):
        if False:
            while True:
                i = 10
        actual_values = []
        for _ in range(len(expected_values)):
            if enable_get_next_as_optional:
                next_element = iterator.get_next_as_optional().get_value()
            else:
                next_element = iterator.get_next()
            computed_value = evaluate_fn([distribute_utils.select_replica(r, next_element) for r in range(len(devices))])
            actual_values.append(computed_value)
        for (expected_value, actual_value) in zip(expected_values, actual_values):
            for (expected, actual) in zip(expected_value, actual_value):
                self.assertAllEqual(expected, actual)

    def _assert_dataset_values_for_loop(self, dataset, expected_values, evaluate_fn, devices):
        if False:
            for i in range(10):
                print('nop')
        actual_values = []
        for x in dataset:
            computed_value = self.evaluate([distribute_utils.select_replica(r, x) for r in range(len(devices))])
            actual_values.append(computed_value)
        for (expected_value, actual_value) in zip(expected_values, actual_values):
            for (expected, actual) in zip(expected_value, actual_value):
                self.assertAllEqual(expected, actual)

    def _test_input_iteration(self, input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, strategy, sess=None, num_replicas_in_sync=None, input_context=None):
        if False:
            while True:
                i = 10
        if iteration_type == 'for_loop' and (not context.executing_eagerly()):
            self.skipTest('unsupported test combination.')
        if api_type == 'wrap_into_iterator' and iteration_type == 'for_loop':
            self.skipTest('unsupported test combination.')
        if api_type == 'wrap_into_iterator' and input_type == 'input_fn':
            self.skipTest('unsupported test combination.')
        devices = nest.flatten([ds for (_, ds) in worker_device_pairs])
        input_workers = input_lib.InputWorkers(worker_device_pairs)
        if api_type == 'wrap_into_iterator':
            iterator = self._wrap_iterator(input_type, dataset_or_input_fn, input_workers, devices, num_replicas_in_sync, strategy, input_context=input_context)
        else:
            dataset = self._wrap_dataset(input_type, dataset_or_input_fn, input_workers, num_replicas_in_sync, strategy, input_context=input_context)
            if ops.executing_eagerly_outside_functions():
                iterator = iter(dataset)
            elif isinstance(dataset, input_lib_v1.DistributedDatasetV1):
                iterator = dataset.make_initializable_iterator()
            else:
                self.skipTest('unsupported test combination')
        if isinstance(iterator, composite_tensor.CompositeTensor):
            nest.assert_same_structure(iterator, iterator._type_spec, expand_composites=True)
        if iteration_type == 'get_next':
            evaluate = lambda x: sess.run(x) if sess else self.evaluate(x)
            if not ops.executing_eagerly_outside_functions():
                evaluate(control_flow_ops.group(iterator.initializer))

            def test_get_next(iterator):
                if False:
                    for i in range(10):
                        print('nop')
                self._assert_iterator_values(iterator, expected_values, evaluate, devices)
                with self.assertRaises(errors.OutOfRangeError):
                    self._assert_iterator_values(iterator, expected_values, evaluate, devices)
                if not ops.executing_eagerly_outside_functions():
                    evaluate(control_flow_ops.group(iterator.initializer))
                elif api_type == 'wrap_into_iterator':
                    self.skipTest('unsupported test combination')
                else:
                    iterator = iter(dataset)
                self._assert_iterator_values(iterator, expected_values, evaluate, devices)

            def test_get_next_as_optional(iterator):
                if False:
                    print('Hello World!')
                self._assert_iterator_values(iterator, expected_values, evaluate, devices, enable_get_next_as_optional=True)
                next_element = iterator.get_next_as_optional()
                self.assertFalse(self.evaluate(next_element.has_value()))
                with self.assertRaises(errors.InvalidArgumentError):
                    self._assert_iterator_values(iterator, [0], evaluate, devices, enable_get_next_as_optional=True)
            test_get_next(iterator)
            if not tf2.enabled():
                return
            elif api_type == 'wrap_into_iterator':
                return
            else:
                iterator = iter(dataset)
            test_get_next_as_optional(iterator)
        if iteration_type == 'for_loop' and context.executing_eagerly():
            self._assert_dataset_values_for_loop(dataset, expected_values, self.evaluate, devices)

    def _create_dataset_or_input_fn(self, input_type, input_fn):
        if False:
            while True:
                i = 10
        if input_type == 'input_fn':
            return input_fn
        else:
            return input_fn(distribute_lib.InputContext())

class DistributedIteratorTest(DistributedIteratorTestBase, parameterized.TestCase):

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu]))
    def testMultiDeviceIterInitialize(self, distribution):
        if False:
            print('Hello World!')
        if tf2.enabled():
            self.skipTest('Only V1 is supported.')
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:CPU:0'])]
        dataset_fn = lambda _: dataset_ops.DatasetV1.range(10)
        input_workers = input_lib.InputWorkers(worker_device_pairs)
        dist_dataset = input_util.get_distributed_dataset(dataset_fn(distribute_lib.InputContext()), input_workers, distribution)
        iterator = dataset_ops.make_one_shot_iterator(dist_dataset)

        @def_function.function
        def init_func_for_iter():
            if False:
                i = 10
                return i + 15
            self.evaluate(iterator.initializer)
        init_func_for_iter()

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu], enable_get_next_as_optional=[True, False]))
    def testOneDeviceCPU(self, input_type, api_type, iteration_type, distribution, enable_get_next_as_optional):
        if False:
            while True:
                i = 10
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        dataset_fn = lambda _: dataset_ops.Dataset.range(10)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        expected_values = [[i] for i in range(10)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['input_fn', 'dataset'], distribution=[strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_one_gpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu]))
    def testAutoShardExplicit(self, input_type, distribution):
        if False:
            print('Hello World!')
        worker_device_pairs = [('/device:CPU:0', distribution.extended.worker_devices)]
        dataset_fn = lambda _: dataset_ops.Dataset.range(10).batch(1)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        input_workers = input_lib.InputWorkers(worker_device_pairs)
        distribution.extended.experimental_enable_get_next_as_optional = True
        dataset = self._wrap_dataset(input_type, dataset_or_input_fn, input_workers, num_replicas_in_sync=None, strategy=distribution)
        dataset1 = input_ops.auto_shard_dataset(dataset, 2, 0)
        iterator = iter(dataset1)
        if len(distribution.extended.worker_devices) == 2:
            expected_values = [[0, 2], [4, 6], [8]]
        else:
            expected_values = [[0], [2], [4], [6], [8]]
        for (element, expected) in zip(iterator, expected_values):
            local = distribution.experimental_local_results(element)
            local_list = array_ops.concat(local, axis=0).numpy().tolist()
            self.assertAllEqual(local_list, expected)
        if len(distribution.extended.worker_devices) == 2:
            expected_values = [[1, 3], [5, 7], [9]]
        else:
            expected_values = [[1], [3], [5], [7], [9]]
        dataset2 = input_ops.auto_shard_dataset(dataset, 2, 1)
        iterator = iter(dataset2)
        for (element, expected) in zip(iterator, expected_values):
            local = distribution.experimental_local_results(element)
            local_list = array_ops.concat(local, axis=0).numpy().tolist()
            self.assertAllEqual(local_list, expected)

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu], enable_get_next_as_optional=[True, False]))
    def testOneDeviceCPUMultiWorker(self, input_type, api_type, iteration_type, distribution, enable_get_next_as_optional):
        if False:
            for i in range(10):
                print('nop')
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        dataset_fn = lambda _: dataset_ops.DatasetV1.range(10)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        expected_values = [[i] for i in range(10)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_gpu_and_cpu], enable_get_next_as_optional=[True, False]))
    def testTwoDevicesOneGPUOneCPU(self, input_type, api_type, iteration_type, distribution, enable_get_next_as_optional):
        if False:
            print('Hello World!')
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:CPU:0'])]
        dataset_fn = lambda _: dataset_ops.Dataset.range(10)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        expected_values = [[i, i + 1] for i in range(0, 10, 2)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.tpu_strategy], enable_get_next_as_optional=[True, False]))
    def testTPU(self, input_type, api_type, iteration_type, distribution, enable_get_next_as_optional):
        if False:
            print('Hello World!')
        worker_device_pairs = collections.OrderedDict()
        for tpu_device in distribution.extended.worker_devices:
            host_device = device_util.get_host_for_device(tpu_device)
            worker_device_pairs.setdefault(host_device, [])
            worker_device_pairs[host_device].append(tpu_device)
        worker_device_pairs = worker_device_pairs.items()
        dataset_fn = lambda _: dataset_ops.Dataset.range(10)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        expected_values = [[i, i + 1] for i in range(0, 10, 2)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_gpu_and_cpu], enable_get_next_as_optional=[True, False]))
    def testTupleDataset(self, input_type, api_type, iteration_type, distribution, enable_get_next_as_optional):
        if False:
            while True:
                i = 10
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:CPU:0'])]

        def dataset_fn(ctx):
            if False:
                print('Hello World!')
            del ctx
            dataset1 = dataset_ops.Dataset.range(10)
            dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x ** 2)
            return dataset_ops.Dataset.zip((dataset1, dataset2))
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        expected_values = [[(i, i ** 2), (i + 1, (i + 1) ** 2)] for i in range(0, 10, 2)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call], enable_get_next_as_optional=[True, False]))
    def testTupleDatasetMultiworker(self, input_type, api_type, iteration_type, distribution, enable_get_next_as_optional):
        if False:
            print('Hello World!')
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:GPU:1'])]

        def dataset_fn(ctx):
            if False:
                while True:
                    i = 10
            del ctx
            dataset1 = dataset_ops.Dataset.range(10)
            dataset2 = dataset_ops.Dataset.range(10).map(lambda x: x ** 2)
            return dataset_ops.Dataset.zip((dataset1, dataset2))
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        expected_values = [[(i, i ** 2), (i + 1, (i + 1) ** 2)] for i in range(0, 10, 2)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testIterableIterator(self, distribution):
        if False:
            i = 10
            return i + 15
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        input_workers = input_lib.InputWorkers(worker_device_pairs)
        dataset = dataset_ops.Dataset.range(10)
        dist_dataset = input_util.get_distributed_dataset(dataset, input_workers, distribution)
        iterator = iter(dist_dataset)
        for (i, element) in enumerate(iterator):
            self.assertAllEqual(distribution.experimental_local_results(element), [i])

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_two_cpus], use_iterator=[False, True]))
    def testIteratorAndDatasetEnumerateError(self, distribution, use_iterator):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(10).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        if use_iterator:
            iterable = iter(dist_dataset)
        else:
            iterable = dist_dataset

        @def_function.function
        def enumerate_fn(iterable):
            if False:
                return 10
            for (_, batch) in enumerate(iterable):
                distribution.experimental_local_results(batch)
        with self.assertRaises(NotImplementedError):
            enumerate_fn(iterable)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_two_cpus]))
    def testIterableIteratorError(self, distribution):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(10).batch(2)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        iterator = iter(dist_dataset)
        with self.assertRaises(ValueError):

            def replica_fn1(iterator):
                if False:
                    return 10
                return next(iterator)
            distribution.run(replica_fn1, args=(iterator,))
        if distribution.num_replicas_in_sync == 1:
            expected_result = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[8, 9]]]
        elif distribution.num_replicas_in_sync == 2:
            expected_result = [[[0], [1]], [[2], [3]], [[4], [5]], [[6], [7]], [[8], [9]]]
        with distribution.scope():

            def replica_fn2(iterator):
                if False:
                    i = 10
                    return i + 15
                return iterator
            result = distribution.run(replica_fn2, args=(next(iterator),))
            self.assertAllEqual(distribution.experimental_local_results(result), expected_result[0])
        iterator = iter(dist_dataset)
        for (i, element) in enumerate(iterator):
            self.assertAllEqual(distribution.experimental_local_results(element), expected_result[i])

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], drop_remainder=[True, False], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_gpu_and_cpu]))
    def testUnevenDatasetBatches(self, input_type, api_type, iteration_type, drop_remainder, distribution):
        if False:
            while True:
                i = 10
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:CPU:0'])]
        dataset_fn = lambda _: dataset_ops.Dataset.range(9).batch(2, drop_remainder=drop_remainder)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        if drop_remainder:
            expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        else:
            expected_values = [[[0, 1], [2, 3]], [[4, 5], [6, 7]], [[8], []]]
        distribution.extended.experimental_enable_get_next_as_optional = True
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], drop_remainder=[True, False], distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu]))
    def testUnevenDatasetBatchesMultiWorker(self, input_type, api_type, iteration_type, drop_remainder, distribution):
        if False:
            i = 10
            return i + 15
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        cr = distribution.cluster_resolver
        self.assertIsNotNone(cr)
        worker_count = multi_worker_util.worker_count(cr.cluster_spec(), cr.task_type)
        id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(), cr.task_type, cr.task_id)

        def dataset_fn(_):
            if False:
                print('Hello World!')
            dataset = dataset_ops.Dataset.range(9)
            if input_type == 'input_fn':
                return dataset.shard(worker_count, id_in_cluster).batch(1)
            else:
                return dataset.batch(2, drop_remainder=drop_remainder)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        if drop_remainder and input_type == 'dataset':
            if id_in_cluster == 0:
                expected_values = [[[0]], [[2]], [[4]], [[6]]]
            else:
                expected_values = [[[1]], [[3]], [[5]], [[7]]]
        elif id_in_cluster == 0:
            expected_values = [[[0]], [[2]], [[4]], [[6]], [[8]]]
        else:
            expected_values = [[[1]], [[3]], [[5]], [[7]], [[]]]
        distribution.extended.experimental_enable_get_next_as_optional = True
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution, num_replicas_in_sync=distribution.num_replicas_in_sync, input_context=distribution.extended._make_input_context())

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['input_fn', 'dataset'], api_type=['wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], drop_remainder=[True, False], distribution=[strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call]))
    def testUnevenDatasetBatchesMultiWorkerFourReplicas(self, input_type, api_type, iteration_type, drop_remainder, distribution):
        if False:
            for i in range(10):
                print('nop')
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:GPU:1'])]
        cr = distribution.cluster_resolver
        self.assertIsNotNone(cr)
        worker_count = multi_worker_util.worker_count(cr.cluster_spec(), cr.task_type)
        id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(), cr.task_type, cr.task_id)

        def dataset_fn(_):
            if False:
                print('Hello World!')
            dataset = dataset_ops.Dataset.range(15)
            if input_type == 'input_fn':
                return dataset.shard(worker_count, id_in_cluster).batch(1)
            else:
                return dataset.batch(4, drop_remainder=drop_remainder)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        if drop_remainder and input_type == 'dataset':
            if id_in_cluster == 0:
                expected_values = [[[0], [2]], [[4], [6]], [[8], [10]]]
            else:
                expected_values = [[[1], [3]], [[5], [7]], [[9], [11]]]
        elif id_in_cluster == 0:
            expected_values = [[[0], [2]], [[4], [6]], [[8], [10]], [[12], [14]]]
        else:
            expected_values = [[[1], [3]], [[5], [7]], [[9], [11]], [[13], []]]
        distribution.extended.experimental_enable_get_next_as_optional = True
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution, num_replicas_in_sync=distribution.num_replicas_in_sync, input_context=distribution.extended._make_input_context())

    @combinations.generate(combinations.combine(mode=['graph', 'eager'], input_type=['dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], num_replicas_in_sync=[None, 2], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_gpu_and_cpu], enable_get_next_as_optional=[True, False]))
    def testBatchSplitting(self, input_type, api_type, iteration_type, num_replicas_in_sync, distribution, enable_get_next_as_optional):
        if False:
            i = 10
            return i + 15
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:CPU:0'])]
        batch_size = 10
        dataset_fn = lambda _: dataset_ops.Dataset.range(100).batch(batch_size)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        updated_batch_size = batch_size // num_replicas_in_sync if num_replicas_in_sync else batch_size
        expected_values = [[range(i, i + updated_batch_size), range(i + updated_batch_size, i + 2 * updated_batch_size)] for i in range(0, 100, updated_batch_size * 2)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution, sess=None, num_replicas_in_sync=num_replicas_in_sync)

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['dataset'], api_type=['wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], num_replicas_in_sync=[None, 2], distribution=[strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call], enable_get_next_as_optional=[True, False]))
    def testBatchSplittingMultiWorker(self, input_type, api_type, iteration_type, num_replicas_in_sync, distribution, enable_get_next_as_optional):
        if False:
            for i in range(10):
                print('nop')
        worker_device_pairs = [('/device:CPU:0', ['/device:GPU:0', '/device:GPU:1'])]
        batch_size = 10
        cr = distribution.cluster_resolver
        self.assertIsNotNone(cr)

        def dataset_fn(_):
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.range(100).batch(batch_size)
            return dataset
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        updated_batch_size = batch_size // num_replicas_in_sync if num_replicas_in_sync else batch_size
        expected_values = [[range(i, i + updated_batch_size), range(i + updated_batch_size, i + 2 * updated_batch_size)] for i in range(0, 100, updated_batch_size * 2)]
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution, sess=None, num_replicas_in_sync=num_replicas_in_sync)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testCacheAcrossIteration(self, distribution):
        if False:
            i = 10
            return i + 15
        if not tf2.enabled():
            self.skipTest('Only V2 is supported.')
        dataset = dataset_ops.Dataset.range(16).shuffle(16).cache().batch(4)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        first_epoch = list((distribution.experimental_local_results(x) for x in dist_dataset))
        second_epoch = list((distribution.experimental_local_results(x) for x in dist_dataset))
        self.assertAllEqual(first_epoch, second_epoch)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu], reshuffle=[True, False]))
    def testShuffleAcrossIterations(self, distribution, reshuffle):
        if False:
            for i in range(10):
                print('nop')
        if not tf2.enabled():
            self.skipTest('Only V2 is supported.')
        dataset = dataset_ops.Dataset.range(12).shuffle(12, reshuffle_each_iteration=reshuffle).batch(4)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        first_epoch = list((distribution.experimental_local_results(x) for x in dist_dataset))
        second_epoch = list((distribution.experimental_local_results(x) for x in dist_dataset))
        if reshuffle:
            self.assertNotAllEqual(first_epoch, second_epoch)
        else:
            self.assertAllEqual(first_epoch, second_epoch)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testGetNextOptionalShapeFinite(self, distribution):
        if False:
            i = 10
            return i + 15
        batch_size = 8
        dataset = dataset_ops.DatasetV2.from_tensor_slices({'feature': array_ops.ones([batch_size, 10]), 'label': array_ops.ones([batch_size])})
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dist_dataset = distribution.experimental_distribute_dataset(dataset)

        @def_function.function
        def train_fn():
            if False:
                i = 10
                return i + 15
            for data in dist_dataset:
                data = nest.map_structure(distribution.experimental_local_results, data)
                feature = data['feature']
                label = data['label']
                for replica_id in range(len(distribution.extended.worker_devices)):
                    self.assertEqual([None, 10], feature[replica_id].shape.as_list())
                    self.assertEqual([None], label[replica_id].shape.as_list())
        train_fn()

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testGetNextOptionalShapeInfinite(self, distribution):
        if False:
            return 10
        batch_size = 8
        dataset = dataset_ops.DatasetV2.from_tensor_slices({'feature': array_ops.ones([batch_size, 10]), 'label': array_ops.ones([batch_size])})
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        per_replica_batch_size = batch_size // distribution.num_replicas_in_sync

        @def_function.function
        def train_fn():
            if False:
                return 10
            data = iter(dist_dataset).get_next_as_optional().get_value()
            data = nest.map_structure(distribution.experimental_local_results, data)
            feature = data['feature']
            label = data['label']
            for replica_id in range(len(distribution.extended.worker_devices)):
                self.assertEqual([per_replica_batch_size, 10], feature[replica_id].shape.as_list())
                self.assertEqual([per_replica_batch_size], label[replica_id].shape.as_list())
        train_fn()

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testGetNextOptionalShapeEmpty(self, distribution):
        if False:
            print('Hello World!')
        batch_size = 8
        dataset = dataset_ops.DatasetV2.from_tensor_slices({'feature': array_ops.ones([batch_size, 10]), 'label': array_ops.ones([batch_size])})
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.repeat()
        dist_dataset = distribution.experimental_distribute_dataset(dataset)
        per_replica_batch_size = batch_size // distribution.num_replicas_in_sync

        @def_function.function
        def train_fn():
            if False:
                return 10
            data = iter(dist_dataset).get_next_as_optional()
            feature_specs = data.element_spec['feature']._component_specs
            value_specs = data.element_spec['label']._component_specs
            if not isinstance(feature_specs, tuple):
                feature_specs = (feature_specs,)
                value_specs = (value_specs,)
            for replica_id in range(len(distribution.extended.worker_devices)):
                self.assertEqual([per_replica_batch_size, 10], feature_specs[replica_id].shape.as_list())
                self.assertEqual([per_replica_batch_size], value_specs[replica_id].shape.as_list())
        train_fn()

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu], input_type=['dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], auto_shard_policy=[AutoShardPolicy.AUTO, AutoShardPolicy.OFF]))
    def testAutoshardingOption(self, distribution, input_type, api_type, iteration_type, auto_shard_policy):
        if False:
            while True:
                i = 10
        cr = distribution.cluster_resolver
        self.assertIsNotNone(cr)
        id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(), cr.task_type, cr.task_id)
        ds_option = options_lib.Options()
        ds_option.experimental_distribute.auto_shard_policy = auto_shard_policy
        dataset_fn = lambda _: dataset_ops.Dataset.range(4).with_options(ds_option)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        if auto_shard_policy == AutoShardPolicy.AUTO:
            if id_in_cluster == 0:
                expected_values = [[0], [2]]
            else:
                expected_values = [[1], [3]]
        else:
            expected_values = [[0], [1], [2], [3]]
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution, input_context=distribution.extended._make_input_context())

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu], input_type=['input_fn'], api_type=['wrap_into_dataset'], iteration_type=['get_next', 'for_loop']))
    def testDifferentDatasetsMultiWorker(self, distribution, input_type, api_type, iteration_type):
        if False:
            for i in range(10):
                print('nop')
        cr = distribution.cluster_resolver
        self.assertIsNotNone(cr)
        id_in_cluster = multi_worker_util.id_in_cluster(cr.cluster_spec(), cr.task_type, cr.task_id)

        def dataset_fn(ctx):
            if False:
                i = 10
                return i + 15
            if ctx.input_pipeline_id == 0:
                return dataset_ops.Dataset.range(8).batch(2)
            else:
                return dataset_ops.Dataset.range(9).batch(2)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        if id_in_cluster == 0:
            expected_values = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[]]]
        else:
            expected_values = [[[0, 1]], [[2, 3]], [[4, 5]], [[6, 7]], [[8]]]
        distribution.extended.experimental_enable_get_next_as_optional = True
        self._test_input_iteration(input_type, api_type, iteration_type, dataset_or_input_fn, worker_device_pairs, expected_values, distribution)

    @combinations.generate(combinations.combine(strategy=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu], mode=['eager']))
    def testLoopOverDatasetInTFFunction(self, strategy):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.range(10).map(lambda x: {'y': math_ops.cast(x, dtypes.float32) ** 2}).batch(4)
        dist_dataset = strategy.experimental_distribute_dataset(dataset)
        with strategy.scope():
            v = variables.Variable(0.0, aggregation=variables.VariableAggregation.SUM)

        @def_function.function
        def iterator_fn(dist_dataset):
            if False:
                print('Hello World!')

            def assign_add_fn(data):
                if False:
                    return 10
                v.assign_add(math_ops.reduce_sum(data['y']))
            for data in dist_dataset:
                strategy.run(assign_add_fn, args=(data,))
        iterator_fn(dist_dataset)
        self.assertEqual(v.numpy(), 285.0)

class DistributedIteratorTensorTypeTest(DistributedIteratorTestBase, parameterized.TestCase):
    """Tests for DistributedDataset with non-dense tensors."""

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_gpu_and_cpu, strategy_combinations.multi_worker_mirrored_2x2_gpu], input_type=['dataset', 'input_fn'], drop_remainder=[False, True], defun_type=['lambda', 'tf_function']))
    def testRaggedSparse(self, distribution, input_type, drop_remainder, defun_type):
        if False:
            print('Hello World!')
        'Test with `RaggedTensor`s and `SparseTensor`s.'
        self.skipTest('b/213596871, b/214574707')
        if not tf2.enabled():
            self.skipTest('Only V2 is supported.')
        defun = {'lambda': lambda f: f, 'tf_function': def_function.function}[defun_type]
        distribution.extended.experimental_enable_get_next_as_optional = True
        global_batch_size = 8

        def dataset_fn(ctx=None):
            if False:
                for i in range(10):
                    print('nop')
            ctx = ctx or distribute_lib.InputContext()
            batch_size = ctx.get_per_replica_batch_size(global_batch_size)
            row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
            ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
            dataset = dataset_ops.DatasetV2.from_tensor_slices({'dense': ragged_tensor.to_tensor(), 'ragged': ragged_tensor, 'sparse': ragged_tensor.to_sparse()})
            dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
            return dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
        dataset_or_input_fn = self._create_dataset_or_input_fn(input_type, dataset_fn)
        dataset = self._wrap_dataset(input_type, dataset_or_input_fn, distribution.extended._input_workers, distribution.num_replicas_in_sync, distribution)
        per_replica_batch = defun(lambda x: next(iter(x)))(dataset)
        self.assertAllEqual(distribute_utils.select_replica(0, per_replica_batch['dense']), [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 2.0, 0.0], [3.0, 3.0, 3.0]])
        self.assertAllEqual(distribute_utils.select_replica(1, per_replica_batch['dense']), [[0.0, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 6.0, 0.0], [7.0, 7.0, 7.0]])
        for i in range(2):
            self.assertLen(distribute_utils.select_replica(i, per_replica_batch['ragged']).values, 6)
            self.assertAllEqual(distribute_utils.select_replica(i, per_replica_batch['ragged']).to_tensor(), distribute_utils.select_replica(i, per_replica_batch['dense']))
            self.assertLen(distribute_utils.select_replica(i, per_replica_batch['sparse']).indices, 6)
            self.assertAllEqual(sparse_ops.sparse_tensor_to_dense(distribute_utils.select_replica(i, per_replica_batch['sparse'])), distribute_utils.select_replica(i, per_replica_batch['dense']))

        def sum_batch(per_replica_features):
            if False:
                return 10
            'Sums the `PerReplica` values in the `per_replica_features` map.'

            def map_fn(per_replica_values):
                if False:
                    print('Hello World!')
                per_replica_sums = distribution.run((lambda x: math_ops.reduce_sum(x.values)) if all(map(sparse_tensor.is_sparse, per_replica_values.values)) else math_ops.reduce_sum, (per_replica_values,))
                return distribution.reduce(reduce_util.ReduceOp.SUM, per_replica_sums, axis=None)
            return nest.map_structure(map_fn, per_replica_features)

        def _reduce(state, batch):
            if False:
                i = 10
                return i + 15
            sums = sum_batch(batch)
            return {name: value + sums[name] for (name, value) in state.items()}

        def sum_for_loop(dataset):
            if False:
                while True:
                    i = 10
            sums = {'dense': 0.0, 'ragged': 0.0, 'sparse': 0.0}
            for batch in dataset:
                sums = _reduce(sums, batch)
            return sums

        def sum_while_loop(iterator, reduce_fn):
            if False:
                for i in range(10):
                    print('nop')
            sums = {'dense': 0.0, 'ragged': 0.0, 'sparse': 0.0}
            while True:
                try:
                    sums = reduce_fn(sums, iterator)
                except (StopIteration, errors.OutOfRangeError):
                    return sums
        while_sums = sum_while_loop(iter(dataset), defun(lambda state, iterator: _reduce(state, next(iterator))))
        self.assertAllEqual(nest.flatten(while_sums), [200.0 if drop_remainder else 310.0] * 3)
        for_sums = defun(sum_for_loop)(dataset)
        expected_for_sum = 200.0
        if not drop_remainder or (defun_type == 'tf_function' and input_type == 'input_fn'):
            expected_for_sum = 310.0
        self.assertAllEqual(nest.flatten(for_sums), [expected_for_sum] * 3)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_gpu_and_cpu, strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu], input_type=['dataset', 'input_fn'], drop_remainder=[False, True], tensor_type=['sparse', 'ragged'], enable_get_next_as_optional=[True, False]))
    def testRaggedSparseGetNextAsOptional(self, distribution, input_type, drop_remainder, tensor_type, enable_get_next_as_optional):
        if False:
            i = 10
            return i + 15
        'Test with `RaggedTensor`s and `SparseTensor`s.'
        if not tf2.enabled():
            self.skipTest('Only V2 is supported.')
        distribution.extended.experimental_enable_get_next_as_optional = enable_get_next_as_optional
        global_batch_size = 8

        def dataset_fn(ctx=None):
            if False:
                i = 10
                return i + 15
            ctx = ctx or distribute_lib.InputContext()
            batch_size = ctx.get_per_replica_batch_size(global_batch_size)
            row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
            ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
            dataset = dataset_ops.DatasetV2.from_tensor_slices({tensor_type: ragged_tensor if tensor_type == 'ragged' else ragged_tensor.to_sparse()})
            dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
            return dataset.batch(batch_size, drop_remainder=drop_remainder)
        if input_type == 'dataset':
            ds = distribution.experimental_distribute_dataset(dataset_fn(distribute_lib.InputContext()))
        else:
            ds = distribution.distribute_datasets_from_function(dataset_fn)
        iterator = iter(ds)
        self.assertEqual(iterator._enable_get_next_as_optional, not drop_remainder and enable_get_next_as_optional)

    @combinations.generate(combinations.combine(tf_api_version=2, mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.central_storage_strategy_with_gpu_and_cpu, strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu], input_type=['dataset', 'input_fn'], drop_remainder=[False, True]))
    def testRaggedSparseGetNextAsOptionalInLoop(self, distribution, input_type, drop_remainder):
        if False:
            i = 10
            return i + 15
        'Test with `RaggedTensor`s and `SparseTensor`s.'
        global_batch_size = 8

        def dataset_fn(ctx=None):
            if False:
                while True:
                    i = 10
            ctx = ctx or distribute_lib.InputContext()
            batch_size = ctx.get_per_replica_batch_size(global_batch_size)
            row_lengths = np.mod(np.arange(20), 4).astype(np.int64)
            ragged_tensor = ragged_tensor_lib.RaggedTensor.from_row_lengths(np.repeat(np.arange(20, dtype=np.float32), row_lengths), row_lengths)
            dataset = dataset_ops.DatasetV2.from_tensor_slices({'dense': ragged_tensor.to_tensor(), 'ragged': ragged_tensor, 'sparse': ragged_tensor.to_sparse()})
            dataset = dataset.batch(batch_size, drop_remainder=drop_remainder)
            return dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)
        if input_type == 'dataset':
            ds = distribution.experimental_distribute_dataset(dataset_fn(distribute_lib.InputContext()))
        else:
            ds = distribution.distribute_datasets_from_function(dataset_fn)

        def sum_batch(per_replica_features):
            if False:
                return 10
            'Sums the `PerReplica` values in the `per_replica_features` map.'

            def map_fn(per_replica_values):
                if False:
                    return 10

                def _sum(value):
                    if False:
                        while True:
                            i = 10
                    if sparse_tensor.is_sparse(value):
                        return math_ops.reduce_sum(value.values)
                    else:
                        return math_ops.reduce_sum(value)
                per_replica_sums = distribution.run(_sum, args=(per_replica_values,))
                return distribution.reduce(reduce_util.ReduceOp.SUM, per_replica_sums, axis=None)
            return nest.map_structure(map_fn, per_replica_features)

        def _reduce(state, batch):
            if False:
                i = 10
                return i + 15
            sums = sum_batch(batch)
            return {name: value + sums[name] for (name, value) in state.items()}

        def sum_while_loop(ds):
            if False:
                for i in range(10):
                    print('nop')
            iterator = iter(ds)
            sums = {'dense': 0.0, 'ragged': 0.0, 'sparse': 0.0}
            try_next = constant_op.constant(True)
            while try_next:
                opt_iterate = iterator.get_next_as_optional()
                if opt_iterate.has_value():
                    sums = _reduce(sums, opt_iterate.get_value())
                else:
                    try_next = False
            return sums
        sums = def_function.function(sum_while_loop)(ds)
        expected_for_sum = 200.0
        if not drop_remainder or input_type == 'input_fn':
            expected_for_sum = 310.0
        self.assertAllEqual(nest.flatten(sums), [expected_for_sum] * 3)

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu]))
    def testMWMSPartialBatch(self, input_type, api_type, iteration_type, distribution):
        if False:
            print('Hello World!')

        def dataset_fn(ctx):
            if False:
                for i in range(10):
                    print('nop')
            del ctx
            dataset = dataset_ops.Dataset.range(12).batch(8)
            options = options_lib.Options()
            options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            return dataset
        dataset = self._create_dataset_or_input_fn(input_type, dataset_fn)
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        expected_values = [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9, 10, 11]]]
        self._test_input_iteration(input_type, api_type, iteration_type, dataset, worker_device_pairs, expected_values, distribution, num_replicas_in_sync=distribution.num_replicas_in_sync, input_context=distribution.extended._make_input_context())

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu]))
    def testMWMSPartialBatchWithLegacyRebatch(self, input_type, api_type, iteration_type, distribution):
        if False:
            i = 10
            return i + 15

        def dataset_fn(ctx):
            if False:
                i = 10
                return i + 15
            del ctx
            batch_sizes = dataset_ops.Dataset.from_tensor_slices([8, 4])
            offsets = dataset_ops.Dataset.from_tensor_slices([0, 8])
            dataset = dataset_ops.Dataset.zip((offsets, batch_sizes))

            def map_fn(offset, batch_size):
                if False:
                    for i in range(10):
                        print('nop')
                return math_ops.range(offset, offset + batch_size)
            dataset = dataset.map(map_fn)
            options = options_lib.Options()
            options.experimental_distribute.auto_shard_policy = AutoShardPolicy.OFF
            dataset = dataset.with_options(options)
            return dataset
        dataset = self._create_dataset_or_input_fn(input_type, dataset_fn)
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        expected_values = [[[0, 1, 2, 3]], [[4, 5, 6, 7]], [[8, 9]], [[10, 11]]]
        self._test_input_iteration(input_type, api_type, iteration_type, dataset, worker_device_pairs, expected_values, distribution, num_replicas_in_sync=distribution.num_replicas_in_sync, input_context=distribution.extended._make_input_context())

    @combinations.generate(combinations.combine(mode=['eager'], input_type=['dataset'], api_type=['wrap_into_iterator', 'wrap_into_dataset'], iteration_type=['get_next', 'for_loop'], distribution=[strategy_combinations.multi_worker_mirrored_2x1_cpu, strategy_combinations.multi_worker_mirrored_2x1_gpu], auto_shard_policy=[AutoShardPolicy.AUTO, AutoShardPolicy.DATA]))
    def testMWMSWithDataSharding(self, input_type, api_type, iteration_type, distribution, auto_shard_policy):
        if False:
            for i in range(10):
                print('nop')

        def dataset_fn(ctx):
            if False:
                while True:
                    i = 10
            del ctx
            dataset = dataset_ops.Dataset.range(8).batch(3)
            options = options_lib.Options()
            options.experimental_distribute.auto_shard_policy = auto_shard_policy
            dataset = dataset.with_options(options)
            return dataset
        dataset = self._create_dataset_or_input_fn(input_type, dataset_fn)
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        cr = distribution.cluster_resolver
        worker_id = multi_worker_util.id_in_cluster(cr.cluster_spec(), cr.task_type, cr.task_id)
        if worker_id == 0:
            expected_values = [[[0, 1]], [[3, 4]], [[6]]]
        elif worker_id == 1:
            expected_values = [[[2]], [[5]], [[7]]]
        self._test_input_iteration(input_type, api_type, iteration_type, dataset, worker_device_pairs, expected_values, distribution, num_replicas_in_sync=distribution.num_replicas_in_sync, input_context=distribution.extended._make_input_context())

@framework_test_util.with_eager_op_as_function
class DistributedIteratorPerDeviceTest(DistributedIteratorTestBase, parameterized.TestCase):
    """Tests for PER_WORKER and PER_REPLICA's InputOptions variants."""

    @combinations.generate(combinations.combine(input_options=[distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=True, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_WORKER), distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=True, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_REPLICA)], mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call, strategy_combinations.mirrored_strategy_with_two_cpus, strategy_combinations.mirrored_strategy_with_gpu_and_cpu]))
    def testDevicePlacementForPerWorkerValuesWithPrefetch(self, distribution, input_options):
        if False:
            print('Hello World!')

        def dataset_fn(input_context):
            if False:
                return 10
            return dataset_ops.Dataset.from_tensor_slices([1, 2, 3, 4])
        ds = distribution.experimental_distribute_datasets_from_function(dataset_fn, input_options)
        for x in ds:
            assert x.values[0].device == distribution.extended.worker_devices[0]
            assert x.values[0].backing_device == distribution.extended.worker_devices[0]
            assert x.values[1].device == distribution.extended.worker_devices[1]
            assert x.values[1].backing_device == distribution.extended.worker_devices[1]

    @combinations.generate(combinations.combine(distribution=[strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call, strategy_combinations.mirrored_strategy_with_two_cpus, strategy_combinations.mirrored_strategy_with_gpu_and_cpu], input_options=[distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=False, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_WORKER)], mode=['eager']))
    def testDevicePlacementForPerWorkerValuesWithoutPrefetch(self, distribution, input_options):
        if False:
            while True:
                i = 10

        def dataset_fn(input_context):
            if False:
                for i in range(10):
                    print('nop')
            return dataset_ops.Dataset.from_tensor_slices(np.full(4, input_context.input_pipeline_id))
        ds = distribution.experimental_distribute_datasets_from_function(dataset_fn, input_options)
        for x in ds:
            x = distribution.run(lambda inputs: inputs, args=(x,))
            assert x.values[0].device == '/job:localhost/replica:0/task:0/device:CPU:0'
            assert x.values[0].backing_device == '/job:localhost/replica:0/task:0/device:CPU:0'
            assert x.values[1].device == '/job:localhost/replica:0/task:0/device:CPU:0'
            assert x.values[1].backing_device == '/job:localhost/replica:0/task:0/device:CPU:0'

    @combinations.generate(combinations.combine(input_options=[distribute_lib.InputOptions(experimental_place_dataset_on_device=True, experimental_fetch_to_device=False, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_WORKER), distribute_lib.InputOptions(experimental_place_dataset_on_device=True, experimental_fetch_to_device=True, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_REPLICA)], mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call, strategy_combinations.mirrored_strategy_with_two_cpus, strategy_combinations.mirrored_strategy_with_gpu_and_cpu]))
    def testDevicePlacementForInvalidCombinations(self, distribution, input_options):
        if False:
            print('Hello World!')

        def dataset_fn(input_context):
            if False:
                for i in range(10):
                    print('nop')
            return dataset_ops.Dataset.from_tensor_slices(np.full(4, input_context.input_pipeline_id))
        with self.assertRaises(ValueError):
            distribution.experimental_distribute_datasets_from_function(dataset_fn, input_options)

    @combinations.generate(combinations.combine(input_options=[distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=False, experimental_per_replica_buffer_size=2), distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=True, experimental_per_replica_buffer_size=2)], mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call, strategy_combinations.mirrored_strategy_with_two_cpus, strategy_combinations.mirrored_strategy_with_gpu_and_cpu]))
    def testPrefetchBufferSizeInputOptions(self, distribution, input_options):
        if False:
            print('Hello World!')

        def dataset_fn(input_context):
            if False:
                return 10
            return dataset_ops.Dataset.from_tensor_slices(np.arange(1, 11).reshape((2, 5)) * (input_context.input_pipeline_id + 1))
        ds = distribution.experimental_distribute_datasets_from_function(dataset_fn, input_options)
        x = next(iter(ds))
        assert np.array_equal(x.values[0].numpy(), np.array([1, 2, 3, 4, 5]))
        assert np.array_equal(x.values[1].numpy(), np.array([6, 7, 8, 9, 10]))

    @combinations.generate(combinations.combine(input_options=[distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=False, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_WORKER), distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=True, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_WORKER)], mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call, strategy_combinations.mirrored_strategy_with_two_cpus, strategy_combinations.mirrored_strategy_with_gpu_and_cpu]))
    def testOutputValuesForPerWorkerInputOptions(self, distribution, input_options):
        if False:
            while True:
                i = 10

        def dataset_fn(input_context):
            if False:
                while True:
                    i = 10
            return dataset_ops.Dataset.from_tensor_slices(np.arange(1, 11).reshape((2, 5)) * (input_context.input_pipeline_id + 1))
        ds = distribution.experimental_distribute_datasets_from_function(dataset_fn, input_options)
        x = next(iter(ds))
        assert np.array_equal(x.values[0].numpy(), np.array([1, 2, 3, 4, 5]))
        assert np.array_equal(x.values[1].numpy(), np.array([6, 7, 8, 9, 10]))

    @combinations.generate(combinations.combine(input_options=[distribute_lib.InputOptions(experimental_place_dataset_on_device=True, experimental_fetch_to_device=False, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_REPLICA), distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=False, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_REPLICA), distribute_lib.InputOptions(experimental_place_dataset_on_device=False, experimental_fetch_to_device=True, experimental_replication_mode=distribute_lib.InputReplicationMode.PER_REPLICA)], mode=['eager'], distribution=[strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.mirrored_strategy_with_two_gpus_no_merge_call, strategy_combinations.mirrored_strategy_with_two_cpus, strategy_combinations.mirrored_strategy_with_gpu_and_cpu]))
    def testOutputValuesForPerReplicaInputOptions(self, distribution, input_options):
        if False:
            while True:
                i = 10

        def dataset_fn(input_context):
            if False:
                print('Hello World!')
            return dataset_ops.Dataset.from_tensor_slices(np.arange(1, 10) * (input_context.input_pipeline_id + 1))
        ds = distribution.experimental_distribute_datasets_from_function(dataset_fn, input_options)
        expected = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9])
        for (i, x) in enumerate(ds):
            assert x.values[0].numpy() == expected[i]
            assert x.values[1].numpy() == expected[i] * 2
            loop_num = i
        assert loop_num == len(expected) - 1

class DistributedIteratorTfDataServiceTest(DistributedIteratorTestBase, parameterized.TestCase):
    """Tests for distributed iterators which read from tf.data service."""

    def setUp(self):
        if False:
            i = 10
            return i + 15
        super(DistributedIteratorTfDataServiceTest, self).setUp()
        self.num_workers = 3
        if combinations.in_main_process():
            self.dispatcher = server_lib.DispatchServer()
            self.workers = []
            for _ in range(self.num_workers):
                self.workers.append(server_lib.WorkerServer(server_lib.WorkerConfig(dispatcher_address=self.dispatcher.target.split('://')[1], heartbeat_interval_ms=100, dispatcher_timeout_ms=1000)))
            combinations.env().tf_data_service_dispatcher = self.dispatcher.target

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testTfDataService(self, distribution):
        if False:
            return 10
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        input_workers = input_lib.InputWorkers(worker_device_pairs)
        dataset = dataset_ops.Dataset.range(1, 50)
        dataset = dataset.apply(data_service_ops._distribute(processing_mode=data_service_ops.ShardingPolicy.OFF, service=combinations.env().tf_data_service_dispatcher, job_name='foo'))
        dist_dataset = input_util.get_distributed_dataset(dataset, input_workers, distribution)
        iterator = iter(dist_dataset)
        results = []
        for element in iterator:
            local_results = distribution.experimental_local_results(element)
            for result in local_results:
                if result.numpy() != 0:
                    results.append(result.numpy())
        self.assertNotEmpty(results)
        gathered = distribution.gather(constant_op.constant(results), axis=0)
        self.assertCountEqual(self.num_workers * list(range(1, 50)), gathered)
        histogram_proto = input_lib._distributed_dataset_initialization_time_milliseconds.get_cell(distribution.__class__.__name__, '1').value()
        self.assertGreater(histogram_proto.num, 0.0)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testDistributeDatasetFromFunction(self, distribution):
        if False:
            print('Hello World!')
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        input_workers = input_lib.InputWorkers(worker_device_pairs)
        input_contexts = []
        num_workers = input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=num_workers))
        dataset = dataset_ops.Dataset.range(1, 50)
        dataset_id = 'dataset_id'
        data_service_ops.register_dataset(service=combinations.env().tf_data_service_dispatcher, dataset=dataset, dataset_id=dataset_id)

        def dataset_fn(input_context):
            if False:
                i = 10
                return i + 15
            del input_context
            return data_service_ops.from_dataset_id(processing_mode=data_service_ops.ShardingPolicy.OFF, service=combinations.env().tf_data_service_dispatcher, dataset_id=dataset_id, element_spec=dataset.element_spec, job_name='shared_job')
        dist_dataset = input_util.get_distributed_datasets_from_function(dataset_fn, input_workers, input_contexts, distribution)
        iterator = iter(dist_dataset)
        results = []
        for element in iterator:
            local_results = distribution.experimental_local_results(element)
            for result in local_results:
                if result.numpy() != 0:
                    results.append(result.numpy())
        self.assertNotEmpty(results)
        gathered = distribution.gather(constant_op.constant(results), axis=0)
        self.assertCountEqual(self.num_workers * list(range(1, 50)), gathered)
        histogram_proto = input_lib._distributed_dataset_from_function_initialization_time_milliseconds.get_cell(distribution.__class__.__name__, '1').value()
        self.assertGreater(histogram_proto.num, 0.0)

    @combinations.generate(combinations.combine(mode=['eager'], distribution=[strategy_combinations.one_device_strategy, strategy_combinations.mirrored_strategy_with_one_cpu, strategy_combinations.mirrored_strategy_with_gpu_and_cpu, strategy_combinations.mirrored_strategy_with_two_gpus, strategy_combinations.tpu_strategy, strategy_combinations.central_storage_strategy_with_two_gpus, strategy_combinations.multi_worker_mirrored_2x2_gpu, strategy_combinations.multi_worker_mirrored_2x2_gpu_no_merge_call, strategy_combinations.multi_worker_mirrored_2x1_cpu]))
    def testDistributeDatasetFromFunctionNested(self, distribution):
        if False:
            for i in range(10):
                print('nop')
        worker_device_pairs = [('/device:CPU:0', ['/device:CPU:0'])]
        input_workers = input_lib.InputWorkers(worker_device_pairs)
        input_contexts = []
        num_workers = input_workers.num_workers
        for i in range(num_workers):
            input_contexts.append(distribute_lib.InputContext(num_input_pipelines=num_workers, input_pipeline_id=i, num_replicas_in_sync=num_workers))

        class InnerType(extension_type.ExtensionType):
            tensor: tensor.Tensor

        class OuterType(extension_type.ExtensionType):
            inner: InnerType

        def dataset_fn(input_context):
            if False:
                return 10
            del input_context

            def data_fn(batch_id) -> OuterType:
                if False:
                    print('Hello World!')
                del batch_id
                return OuterType(inner=InnerType(tensor=constant_op.constant([[0.0, 1.0], [2.0, 3.0]])))
            return dataset_ops.Dataset.range(1, 10).map(data_fn)
        dist_dataset = input_util.get_distributed_datasets_from_function(dataset_fn, input_workers, input_contexts, distribution)
        iterator = iter(dist_dataset)
        results = []
        for element in iterator:
            local_results = distribution.experimental_local_results(element)
            for result in local_results:
                results.append(result)
        expect_component = OuterType(inner=InnerType(tensor=constant_op.constant([[0.0, 1.0], [2.0, 3.0]])))
        self.assertCountEqual(num_workers * [expect_component for _ in range(1, 10)], results)
if __name__ == '__main__':
    test_util.main()