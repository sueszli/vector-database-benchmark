"""Tests for `tf.data.Iterator` using distributed sessions."""
from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import lookup_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import string_ops
from tensorflow.python.platform import test

class IteratorClusterTest(test.TestCase, parameterized.TestCase):

    @combinations.generate(test_base.graph_only_combinations())
    def testRemoteIteratorWithoutRemoteCallFail(self):
        if False:
            while True:
                i = 10
        worker_config = config_pb2.ConfigProto()
        worker_config.device_count['CPU'] = 2
        (worker, _) = test_util.create_local_cluster(1, 1, worker_config=worker_config)
        with ops.device('/job:worker/replica:0/task:0/cpu:1'):
            dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
            iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
            iterator_3_handle = iterator_3.string_handle()
        with ops.device('/job:worker/replica:0/task:0/cpu:0'):
            remote_it = iterator_ops.Iterator.from_string_handle(iterator_3_handle, dataset_ops.get_legacy_output_types(dataset_3), dataset_ops.get_legacy_output_shapes(dataset_3))
            get_next_op = remote_it.get_next()
        with session.Session(worker[0].target) as sess:
            with self.assertRaises(errors.InvalidArgumentError):
                sess.run(get_next_op)

    def _testRemoteIteratorHelper(self, device0, device1, target):
        if False:
            return 10
        with ops.device(device1):
            dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
            iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
            iterator_3_handle = iterator_3.string_handle()

        @function.Defun(dtypes.string)
        def _remote_fn(h):
            if False:
                while True:
                    i = 10
            remote_iterator = iterator_ops.Iterator.from_string_handle(h, dataset_ops.get_legacy_output_types(dataset_3), dataset_ops.get_legacy_output_shapes(dataset_3))
            return remote_iterator.get_next()
        with ops.device(device0):
            target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
            remote_op = functional_ops.remote_call(args=[iterator_3_handle], Tout=[dtypes.int32], f=_remote_fn, target=target_placeholder)
        with session.Session(target) as sess:
            elem = sess.run(remote_op, feed_dict={target_placeholder: device1})
            self.assertEqual(elem, [1])
            with self.assertRaises(errors.InvalidArgumentError):
                sess.run(remote_op, feed_dict={target_placeholder: device0})
            elem = sess.run(iterator_3.get_next())
            self.assertEqual(elem, [2])
            elem = sess.run(remote_op, feed_dict={target_placeholder: device1})
            self.assertEqual(elem, [3])
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(remote_op, feed_dict={target_placeholder: device1})

    @combinations.generate(test_base.graph_only_combinations())
    def testRemoteIteratorUsingRemoteCallOp(self):
        if False:
            return 10
        worker_config = config_pb2.ConfigProto()
        worker_config.device_count['CPU'] = 2
        (worker, _) = test_util.create_local_cluster(1, 1, worker_config=worker_config)
        self._testRemoteIteratorHelper('/job:worker/replica:0/task:0/cpu:0', '/job:worker/replica:0/task:0/cpu:1', worker[0].target)

    @combinations.generate(test_base.graph_only_combinations())
    def testRemoteIteratorUsingRemoteCallOpCrossProcess(self):
        if False:
            print('Hello World!')
        (workers, _) = test_util.create_local_cluster(2, 1)
        self._testRemoteIteratorHelper('/job:worker/replica:0/task:0/cpu:0', '/job:worker/replica:0/task:1/cpu:0', workers[0].target)

    @combinations.generate(test_base.graph_only_combinations())
    def testCaptureHashTableInSharedIterator(self):
        if False:
            for i in range(10):
                print('nop')
        (worker, _) = test_util.create_local_cluster(1, 1)
        default_val = -1
        keys = constant_op.constant(['brain', 'salad', 'surgery'])
        values = constant_op.constant([0, 1, 2], dtypes.int64)
        table = lookup_ops.StaticHashTableV1(lookup_ops.KeyValueTensorInitializer(keys, values), default_val)
        input_sentences = dataset_ops.Dataset.from_tensor_slices(['brain brain tank salad surgery', 'surgery brain'])
        dataset = input_sentences.map(lambda x: string_ops.string_split([x]).values).map(table.lookup)
        iterator = dataset_ops.make_initializable_iterator(dataset, shared_name='shared_iterator')
        init_op = iterator.initializer
        get_next = iterator.get_next()
        with session.Session(worker[0].target) as sess:
            sess.run(table.initializer)
            sess.run(init_op)
            self.assertAllEqual([0, 0, -1, 1, 2], sess.run(get_next))
        with session.Session(worker[0].target) as sess:
            self.assertAllEqual([2, 0], sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    @combinations.generate(test_base.graph_only_combinations())
    def testImplicitDisposeParallelMapDataset(self):
        if False:
            for i in range(10):
                print('nop')
        (worker, _) = test_util.create_local_cluster(1, 1)
        components = (np.arange(1000), np.array([[1, 2, 3]]) * np.arange(1000)[:, np.newaxis], np.array(37.0) * np.arange(1000))

        def _map_fn(x, y, z):
            if False:
                i = 10
                return i + 15
            return (math_ops.square(x), math_ops.square(y), math_ops.square(z))
        dataset = dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn).repeat(None).prefetch(10000)
        iterator = dataset_ops.make_initializable_iterator(dataset)
        init_op = iterator.initializer
        get_next = iterator.get_next()
        with session.Session(worker[0].target) as sess:
            sess.run(init_op)
            for _ in range(3):
                sess.run(get_next)
if __name__ == '__main__':
    test.main()