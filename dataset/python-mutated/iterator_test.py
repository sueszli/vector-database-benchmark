"""Tests for `tf.data.Iterator`."""
import warnings
from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import cluster_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.data.kernel_tests import test_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import from_generator_op
from tensorflow.python.data.ops import iterator_ops
from tensorflow.python.data.util import structure
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import combinations
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import sparse_tensor
from tensorflow.python.framework import tensor
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import functional_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import parsing_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib
from tensorflow.python.util import compat

@test_util.with_eager_op_as_function
class IteratorTest(test_base.DatasetTestBase, parameterized.TestCase):

    @combinations.generate(test_base.graph_only_combinations())
    def testNoGradients(self):
        if False:
            i = 10
            return i + 15
        component = constant_op.constant([1.0])
        side = constant_op.constant(0.0)
        add = lambda x: x + side
        dataset = dataset_ops.Dataset.from_tensor_slices(component).map(add)
        value = dataset_ops.make_one_shot_iterator(dataset).get_next()
        self.assertIsNone(gradients_impl.gradients(value, component)[0])
        self.assertIsNone(gradients_impl.gradients(value, side)[0])
        self.assertIsNone(gradients_impl.gradients(value, [component, side])[0])

    @combinations.generate(test_base.graph_only_combinations())
    def testCapturingStateInOneShotRaisesException(self):
        if False:
            return 10
        var = variables.Variable(37.0, name='myvar')
        dataset = dataset_ops.Dataset.from_tensor_slices([0.0, 1.0, 2.0]).map(lambda x: x + var)
        with self.assertRaisesRegex(ValueError, 'A likely cause of this error is that the dataset for which you are calling `make_one_shot_iterator\\(\\)` captures a stateful object, such as a `tf.Variable` or `tf.lookup.StaticHashTable`, which is not supported. Use `make_initializable_iterator\\(\\)` instead.'):
            dataset_ops.make_one_shot_iterator(dataset)

    @combinations.generate(test_base.graph_only_combinations())
    def testOneShotIterator(self):
        if False:
            return 10
        components = (np.arange(7), np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis], np.array(37.0) * np.arange(7))

        def _map_fn(x, y, z):
            if False:
                while True:
                    i = 10
            return (math_ops.square(x), math_ops.square(y), math_ops.square(z))
        iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn).repeat(14))
        get_next = iterator.get_next()
        self.assertEqual([c.shape[1:] for c in components], [t.shape for t in get_next])
        with self.cached_session() as sess:
            for _ in range(14):
                for i in range(7):
                    result = sess.run(get_next)
                    for (component, result_component) in zip(components, result):
                        self.assertAllEqual(component[i] ** 2, result_component)
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    @combinations.generate(test_base.graph_only_combinations())
    def testOneShotIteratorCaptureByValue(self):
        if False:
            print('Hello World!')
        components = (np.arange(7), np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis], np.array(37.0) * np.arange(7))
        tensor_components = tuple([ops.convert_to_tensor(c) for c in components])

        def _map_fn(x, y, z):
            if False:
                i = 10
                return i + 15
            return (math_ops.square(x), math_ops.square(y), math_ops.square(z))
        iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.from_tensor_slices(tensor_components).map(_map_fn).repeat(14))
        get_next = iterator.get_next()
        self.assertEqual([c.shape[1:] for c in components], [t.shape for t in get_next])
        with self.cached_session() as sess:
            for _ in range(14):
                for i in range(7):
                    result = sess.run(get_next)
                    for (component, result_component) in zip(components, result):
                        self.assertAllEqual(component[i] ** 2, result_component)
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    @combinations.generate(test_base.default_test_combinations())
    def testOneShotIteratorInsideContainer(self):
        if False:
            return 10
        components = (np.arange(7), np.array([[1, 2, 3]]) * np.arange(7)[:, np.newaxis], np.array(37.0) * np.arange(7))

        def within_container():
            if False:
                print('Hello World!')

            def _map_fn(x, y, z):
                if False:
                    i = 10
                    return i + 15
                return (math_ops.square(x), math_ops.square(y), math_ops.square(z))
            iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.from_tensor_slices(components).map(_map_fn).repeat(14))
            return iterator.get_next()
        server = server_lib.Server.create_local_server()
        for j in range(2):
            with session.Session(server.target) as sess:
                cname = 'iteration%d' % j
                with ops.container(cname):
                    get_next = within_container()
                for _ in range(14):
                    for i in range(7):
                        result = sess.run(get_next)
                        for (component, result_component) in zip(components, result):
                            self.assertAllEqual(component[i] ** 2, result_component)
                with self.assertRaises(errors.OutOfRangeError):
                    sess.run(get_next)

    @combinations.generate(test_base.graph_only_combinations())
    def testOneShotIteratorNonBlocking(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensors([1, 2, 3]).map(lambda x: x * x)
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()
        config = config_pb2.ConfigProto(inter_op_parallelism_threads=1, use_per_session_threads=True)
        with session.Session(config=config) as sess:
            self.assertAllEqual([1, 4, 9], sess.run(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element)
        with session.Session(config=config) as sess:
            results = []

            def consumer_thread():
                if False:
                    while True:
                        i = 10
                try:
                    results.append(sess.run(next_element))
                except errors.OutOfRangeError:
                    results.append(None)
            num_threads = 8
            threads = [self.checkedThread(consumer_thread) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertLen(results, num_threads)
            self.assertLen([None for r in results if r is None], num_threads - 1)
            self.assertAllEqual([[1, 4, 9]], [r for r in results if r is not None])

    @combinations.generate(test_base.graph_only_combinations())
    def testOneShotIteratorInitializerFails(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.from_tensors(array_ops.gather([0], [4]))
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        next_element = iterator.get_next()
        with self.cached_session() as sess:
            with self.assertRaisesRegex(errors.InvalidArgumentError, ''):
                sess.run(next_element)
            with self.assertRaisesRegex(errors.InvalidArgumentError, ''):
                sess.run(next_element)
        with self.cached_session() as sess:

            def consumer_thread():
                if False:
                    return 10
                with self.assertRaisesRegex(errors.InvalidArgumentError, ''):
                    sess.run(next_element)
            num_threads = 8
            threads = [self.checkedThread(consumer_thread) for _ in range(num_threads)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

    @combinations.generate(test_base.default_test_combinations())
    def testOneShotIteratorEmptyDataset(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.range(0)
        iterator = dataset_ops.make_one_shot_iterator(dataset)
        with self.assertRaises(errors.OutOfRangeError):
            self.evaluate(iterator.get_next())

    @combinations.generate(test_base.graph_only_combinations())
    def testSimpleSharedResource(self):
        if False:
            for i in range(10):
                print('nop')
        components = (np.array(1, dtype=np.int64), np.array([1, 2, 3], dtype=np.int64), np.array(37.0, dtype=np.float64))
        server = server_lib.Server.create_local_server()
        with ops.Graph().as_default():
            iterator = dataset_ops.make_initializable_iterator(dataset_ops.Dataset.from_tensors(components).map(lambda x, y, z: (x, y, z)), shared_name='shared_iterator')
            init_op = iterator.initializer
            get_next = iterator.get_next()
            with session.Session(server.target) as sess:
                sess.run(init_op)
                results = sess.run(get_next)
                for (component, result_component) in zip(components, results):
                    self.assertAllEqual(component, result_component)
                with self.assertRaises(errors.OutOfRangeError):
                    sess.run(get_next)
                sess.run(init_op)
        with ops.Graph().as_default():
            iterator = iterator_ops.Iterator.from_structure(shared_name='shared_iterator', output_types=(dtypes.int64, dtypes.int64, dtypes.float64), output_shapes=([], [3], []))
            get_next = iterator.get_next()
            with session.Session(server.target) as sess:
                results = sess.run(get_next)
                for (component, result_component) in zip(components, results):
                    self.assertAllEqual(component, result_component)
                with self.assertRaises(errors.OutOfRangeError):
                    sess.run(get_next)

    @combinations.generate(test_base.graph_only_combinations())
    def testNotInitializedError(self):
        if False:
            while True:
                i = 10
        components = (np.array(1), np.array([1, 2, 3]), np.array(37.0))
        iterator = dataset_ops.make_initializable_iterator(dataset_ops.Dataset.from_tensors(components))
        get_next = iterator.get_next()
        with self.cached_session() as sess:
            with self.assertRaisesRegex(errors.FailedPreconditionError, 'iterator has not been initialized'):
                sess.run(get_next)

    @combinations.generate(test_base.graph_only_combinations())
    def testReinitializableIterator(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_3 = dataset_ops.Dataset.from_tensors(constant_op.constant([1, 2, 3]))
        dataset_4 = dataset_ops.Dataset.from_tensors(constant_op.constant([4, 5, 6, 7]))
        iterator = iterator_ops.Iterator.from_structure(dataset_ops.get_legacy_output_types(dataset_3), [None])
        dataset_3_init_op = iterator.make_initializer(dataset_3)
        dataset_4_init_op = iterator.make_initializer(dataset_4)
        get_next = iterator.get_next()
        self.assertEqual(dataset_ops.get_legacy_output_types(dataset_3), dataset_ops.get_legacy_output_types(iterator))
        self.assertEqual(dataset_ops.get_legacy_output_types(dataset_4), dataset_ops.get_legacy_output_types(iterator))
        self.assertEqual([None], dataset_ops.get_legacy_output_shapes(iterator).as_list())
        with self.cached_session() as sess:
            with self.assertRaises(errors.FailedPreconditionError):
                sess.run(get_next)
            sess.run(dataset_3_init_op)
            self.assertAllEqual([1, 2, 3], sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)
            sess.run(dataset_4_init_op)
            self.assertAllEqual([4, 5, 6, 7], sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)
            sess.run(dataset_3_init_op)
            self.assertAllEqual([1, 2, 3], sess.run(get_next))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(get_next)

    @combinations.generate(test_base.graph_only_combinations())
    def testReinitializableIteratorWithFunctions(self):
        if False:
            while True:
                i = 10

        def g():
            if False:
                return 10
            for i in range(10):
                yield i
        iterator = iterator_ops.Iterator.from_structure(dtypes.int64, [])
        next_element = iterator.get_next()
        with self.cached_session() as sess:
            dataset_1 = dataset_ops.Dataset.from_generator(g, output_types=dtypes.int64)
            sess.run(iterator.make_initializer(dataset_1))
            for expected in range(10):
                self.assertEqual(expected, sess.run(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element)
            dataset_2 = dataset_ops.Dataset.from_generator(g, output_types=dtypes.int64)
            sess.run(iterator.make_initializer(dataset_2))
            for expected in range(10):
                self.assertEqual(expected, sess.run(next_element))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element)

    @combinations.generate(test_base.default_test_combinations())
    def testReinitializableIteratorStaticErrors(self):
        if False:
            return 10
        with self.assertRaises(TypeError):
            iterator = iterator_ops.Iterator.from_structure((dtypes.int64, dtypes.float64), [None])
        iterator = iterator_ops.Iterator.from_structure((dtypes.int64, dtypes.float64))
        with self.assertRaisesRegex(ValueError, "The two structures don't have the same nested structure."):
            iterator.make_initializer(dataset_ops.Dataset.from_tensors(((constant_op.constant([1, 2, 3], dtype=dtypes.int64),), (constant_op.constant([4.0, 5.0, 6.0, 7.0], dtype=dtypes.float64),))))
        with self.assertRaisesRegex(TypeError, 'Expected output types \\(tf.int64, tf.float64\\) but got dataset with output types \\(tf.int32, tf.float32\\).'):
            iterator.make_initializer(dataset_ops.Dataset.from_tensors((constant_op.constant([1, 2, 3], dtype=dtypes.int32), constant_op.constant([4.0, 5.0, 6.0, 7.0], dtype=dtypes.float32))))
        iterator = iterator_ops.Iterator.from_structure((dtypes.int64, dtypes.float64), ([None], []))
        with self.assertRaisesRegex(TypeError, 'Expected output shapes compatible with .* but got dataset with output shapes.*'):
            iterator.make_initializer(dataset_ops.Dataset.from_tensors((constant_op.constant([1, 2, 3], dtype=dtypes.int64), constant_op.constant([4.0, 5.0, 6.0, 7.0], dtype=dtypes.float64))))

    @combinations.generate(test_base.default_test_combinations())
    def testReinitializableIteratorEmptyDataset(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.range(0)
        iterator = iterator_ops.Iterator.from_structure(dataset_ops.get_legacy_output_types(dataset), [])
        init_op = iterator.make_initializer(dataset)
        with self.cached_session() as sess:
            sess.run(init_op)
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(iterator.get_next())

    @combinations.generate(test_base.graph_only_combinations())
    def testIteratorStringHandle(self):
        if False:
            return 10
        dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
        dataset_4 = dataset_ops.Dataset.from_tensor_slices([10, 20, 30, 40])
        iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
        iterator_4 = dataset_ops.make_one_shot_iterator(dataset_4)
        handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])
        feedable_iterator = iterator_ops.Iterator.from_string_handle(handle_placeholder, dataset_ops.get_legacy_output_types(dataset_3), dataset_ops.get_legacy_output_shapes(dataset_3))
        next_element = feedable_iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(dataset_3), dataset_ops.get_structure(feedable_iterator)))
        with self.cached_session() as sess:
            iterator_3_handle = sess.run(iterator_3.string_handle())
            iterator_4_handle = sess.run(iterator_4.string_handle())
            self.assertEqual(10, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            self.assertEqual(1, sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle}))
            self.assertEqual(20, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            self.assertEqual(2, sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle}))
            self.assertEqual(30, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            self.assertEqual(3, sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle}))
            self.assertEqual(40, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle})
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle})

    @combinations.generate(test_base.graph_only_combinations())
    def testIteratorStringHandleFuture(self):
        if False:
            for i in range(10):
                print('nop')
        dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
        dataset_4 = dataset_ops.Dataset.from_tensor_slices([10, 20, 30, 40])
        iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
        iterator_4 = dataset_ops.make_one_shot_iterator(dataset_4)
        handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])
        feedable_iterator = iterator_ops.Iterator.from_string_handle(handle_placeholder, dataset_ops.get_legacy_output_types(dataset_3), dataset_ops.get_legacy_output_shapes(dataset_3))
        next_element = feedable_iterator.get_next()
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(dataset_3), dataset_ops.get_structure(feedable_iterator)))
        with self.cached_session() as sess:
            iterator_3_handle = sess.run(iterator_3.string_handle())
            iterator_4_handle = sess.run(iterator_4.string_handle())
            self.assertEqual(10, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            self.assertEqual(1, sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle}))
            self.assertEqual(20, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            self.assertEqual(2, sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle}))
            self.assertEqual(30, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            self.assertEqual(3, sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle}))
            self.assertEqual(40, sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle}))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element, feed_dict={handle_placeholder: iterator_3_handle})
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(next_element, feed_dict={handle_placeholder: iterator_4_handle})

    @combinations.generate(test_base.graph_only_combinations())
    def testIteratorStringHandleReuseTensorObject(self):
        if False:
            while True:
                i = 10
        dataset = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
        one_shot_iterator = dataset_ops.make_one_shot_iterator(dataset)
        initializable_iterator = dataset_ops.make_initializable_iterator(dataset)
        structure_iterator = iterator_ops.Iterator.from_structure(dataset_ops.get_legacy_output_types(dataset))
        created_ops = len(ops.get_default_graph().get_operations())
        self.assertIs(one_shot_iterator.string_handle(), one_shot_iterator.string_handle())
        self.assertIs(initializable_iterator.string_handle(), initializable_iterator.string_handle())
        self.assertIs(structure_iterator.string_handle(), structure_iterator.string_handle())
        self.assertLen(ops.get_default_graph().get_operations(), created_ops)
        handle_with_name = one_shot_iterator.string_handle(name='foo')
        self.assertEqual('foo', handle_with_name.op.name)
        self.assertIsNot(one_shot_iterator.string_handle(), handle_with_name)
        handle_with_same_name = one_shot_iterator.string_handle(name='foo')
        self.assertEqual('foo_1', handle_with_same_name.op.name)
        self.assertIsNot(handle_with_name, handle_with_same_name)

    @combinations.generate(test_base.graph_only_combinations())
    def testIteratorStringHandleError(self):
        if False:
            i = 10
            return i + 15
        dataset_int_scalar = dataset_ops.Dataset.from_tensor_slices([1, 2, 3]).repeat()
        dataset_float_vector = dataset_ops.Dataset.from_tensors([1.0, 2.0, 3.0])
        handle_placeholder = array_ops.placeholder(dtypes.string, shape=[])
        feedable_int_scalar = iterator_ops.Iterator.from_string_handle(handle_placeholder, dtypes.int32, [])
        feedable_int_vector = iterator_ops.Iterator.from_string_handle(handle_placeholder, dtypes.int32, [None])
        feedable_int_any = iterator_ops.Iterator.from_string_handle(handle_placeholder, dtypes.int32)
        with self.cached_session() as sess:
            handle_int_scalar = sess.run(dataset_ops.make_one_shot_iterator(dataset_int_scalar).string_handle())
            handle_float_vector = sess.run(dataset_ops.make_one_shot_iterator(dataset_float_vector).string_handle())
            self.assertEqual(1, sess.run(feedable_int_scalar.get_next(), feed_dict={handle_placeholder: handle_int_scalar}))
            self.assertEqual(2, sess.run(feedable_int_any.get_next(), feed_dict={handle_placeholder: handle_int_scalar}))
            with self.assertRaises(errors.InvalidArgumentError):
                print(sess.run(feedable_int_vector.get_next(), feed_dict={handle_placeholder: handle_int_scalar}))
            with self.assertRaises(errors.InvalidArgumentError):
                print(sess.run(feedable_int_vector.get_next(), feed_dict={handle_placeholder: handle_float_vector}))

    @combinations.generate(test_base.graph_only_combinations())
    def testRemoteIteratorUsingRemoteCallOpDirectSession(self):
        if False:
            i = 10
            return i + 15
        worker_config = config_pb2.ConfigProto()
        worker_config.device_count['CPU'] = 3
        with ops.device('/job:localhost/replica:0/task:0/cpu:1'):
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
        with ops.device('/job:localhost/replica:0/task:0/cpu:0'):
            target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
            remote_op = functional_ops.remote_call(args=[iterator_3_handle], Tout=[dtypes.int32], f=_remote_fn, target=target_placeholder)
        with self.session(config=worker_config) as sess:
            elem = sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:1'})
            self.assertEqual(elem, [1])
            with self.assertRaises(errors.InvalidArgumentError):
                sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:2'})
            elem = sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:1'})
            self.assertEqual(elem, [2])
            elem = sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:1'})
            self.assertEqual(elem, [3])
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:1'})

    @combinations.generate(test_base.graph_only_combinations())
    def testRemoteIteratorUsingRemoteCallOpMultiWorkers(self):
        if False:
            return 10
        s1 = server_lib.Server.create_local_server()
        s2 = server_lib.Server.create_local_server()
        s3 = server_lib.Server.create_local_server()
        cluster_def = cluster_pb2.ClusterDef()
        workers = cluster_def.job.add()
        workers.name = 'worker'
        workers.tasks[0] = s1.target[len('grpc://'):]
        workers.tasks[1] = s2.target[len('grpc://'):]
        client = cluster_def.job.add()
        client.name = 'client'
        client.tasks[0] = s3.target[len('grpc://'):]
        config = config_pb2.ConfigProto(cluster_def=cluster_def)
        worker_devices = ['/job:worker/replica:0/task:%d/cpu:0' % i for i in range(2)]
        itr_handles = []
        for device in worker_devices:
            with ops.device(device):
                src = dataset_ops.Dataset.from_tensor_slices([device])
                itr = dataset_ops.make_one_shot_iterator(src)
                itr_handles.append(itr.string_handle())
        targets = dataset_ops.Dataset.from_tensor_slices(worker_devices)
        handles = dataset_ops.Dataset.from_tensor_slices(itr_handles)

        @function.Defun(dtypes.string)
        def loading_func(h):
            if False:
                i = 10
                return i + 15
            remote_itr = iterator_ops.Iterator.from_string_handle(h, dataset_ops.get_legacy_output_types(itr), dataset_ops.get_legacy_output_shapes(itr))
            return remote_itr.get_next()

        def map_fn(target, handle):
            if False:
                return 10
            return functional_ops.remote_call(args=[handle], Tout=[dtypes.string], f=loading_func, target=target)
        with ops.device('/job:client'):
            client_dataset = dataset_ops.Dataset.zip((targets, handles)).map(map_fn)
            itr = dataset_ops.make_initializable_iterator(client_dataset)
            n = itr.get_next()
        with session.Session(s3.target, config=config) as sess:
            sess.run(itr.initializer)
            expected_values = worker_devices
            for expected in expected_values:
                self.assertEqual((compat.as_bytes(expected),), sess.run(n))
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(n)

    @combinations.generate(test_base.graph_only_combinations())
    def testRemoteIteratorUsingRemoteCallOpDirectSessionGPUCPU(self):
        if False:
            print('Hello World!')
        if not test_util.is_gpu_available():
            self.skipTest('No GPU available')
        with ops.device('/job:localhost/replica:0/task:0/cpu:0'):
            dataset_3 = dataset_ops.Dataset.from_tensor_slices([1, 2, 3])
            iterator_3 = dataset_ops.make_one_shot_iterator(dataset_3)
            iterator_3_handle = iterator_3.string_handle()

        def _encode_raw(byte_array):
            if False:
                i = 10
                return i + 15
            return bytes(bytearray(byte_array))

        @function.Defun(dtypes.uint8)
        def _remote_fn(h):
            if False:
                for i in range(10):
                    print('nop')
            handle = script_ops.py_func(_encode_raw, [h], dtypes.string)
            remote_iterator = iterator_ops.Iterator.from_string_handle(handle, dataset_ops.get_legacy_output_types(dataset_3), dataset_ops.get_legacy_output_shapes(dataset_3))
            return remote_iterator.get_next()
        with ops.device('/job:localhost/replica:0/task:0/device:GPU:0'):
            target_placeholder = array_ops.placeholder(dtypes.string, shape=[])
            iterator_3_handle_uint8 = parsing_ops.decode_raw(input_bytes=iterator_3_handle, out_type=dtypes.uint8)
            remote_op = functional_ops.remote_call(args=[iterator_3_handle_uint8], Tout=[dtypes.int32], f=_remote_fn, target=target_placeholder)
        with self.cached_session() as sess:
            elem = sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:0'})
            self.assertEqual(elem, [1])
            elem = sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:0'})
            self.assertEqual(elem, [2])
            elem = sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:0'})
            self.assertEqual(elem, [3])
            with self.assertRaises(errors.OutOfRangeError):
                sess.run(remote_op, feed_dict={target_placeholder: '/job:localhost/replica:0/task:0/cpu:0'})

    @combinations.generate(test_base.graph_only_combinations())
    def testRepeatedGetNextWarning(self):
        if False:
            for i in range(10):
                print('nop')
        iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.range(10))
        warnings.simplefilter('always')
        with warnings.catch_warnings(record=True) as w:
            for _ in range(100):
                iterator.get_next()
        self.assertLen(w, 100 - iterator_ops.GET_NEXT_CALL_WARNING_THRESHOLD)
        for warning in w:
            self.assertIn(iterator_ops.GET_NEXT_CALL_WARNING_MESSAGE, str(warning.message))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(expected_element_structure=tensor.TensorSpec([], dtypes.float32), expected_output_classes=tensor.Tensor, expected_output_types=dtypes.float32, expected_output_shapes=[[]])))
    def testTensorIteratorStructure(self, expected_element_structure, expected_output_classes, expected_output_types, expected_output_shapes):
        if False:
            print('Hello World!')
        tf_value_fn = lambda : constant_op.constant(37.0)
        tf_value = tf_value_fn()
        iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.from_tensors(tf_value))
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(iterator), expected_element_structure))
        self.assertEqual(expected_output_classes, dataset_ops.get_legacy_output_classes(iterator))
        self.assertEqual(expected_output_types, dataset_ops.get_legacy_output_types(iterator))
        self.assertEqual(expected_output_shapes, dataset_ops.get_legacy_output_shapes(iterator))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(expected_element_structure=sparse_tensor.SparseTensorSpec([1], dtypes.int32), expected_output_classes=sparse_tensor.SparseTensor, expected_output_types=dtypes.int32, expected_output_shapes=[[1]])))
    def testSparseTensorIteratorStructure(self, expected_element_structure, expected_output_classes, expected_output_types, expected_output_shapes):
        if False:
            return 10

        def tf_value_fn():
            if False:
                while True:
                    i = 10
            return sparse_tensor.SparseTensor(indices=[[0]], values=constant_op.constant([0], dtype=dtypes.int32), dense_shape=[1])
        tf_value = tf_value_fn()
        iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.from_tensors(tf_value))
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(iterator), expected_element_structure))
        self.assertEqual(expected_output_classes, dataset_ops.get_legacy_output_classes(iterator))
        self.assertEqual(expected_output_types, dataset_ops.get_legacy_output_types(iterator))
        self.assertEqual(expected_output_shapes, dataset_ops.get_legacy_output_shapes(iterator))

    @combinations.generate(combinations.times(test_base.default_test_combinations(), combinations.combine(expected_element_structure={'a': tensor.TensorSpec([], dtypes.float32), 'b': (tensor.TensorSpec([1], dtypes.string), tensor.TensorSpec([], dtypes.string))}, expected_output_classes={'a': tensor.Tensor, 'b': (tensor.Tensor, tensor.Tensor)}, expected_output_types={'a': dtypes.float32, 'b': (dtypes.string, dtypes.string)}, expected_output_shapes={'a': [], 'b': ([1], [])})))
    def testNestedTensorIteratorStructure(self, expected_element_structure, expected_output_classes, expected_output_types, expected_output_shapes):
        if False:
            for i in range(10):
                print('nop')

        def tf_value_fn():
            if False:
                i = 10
                return i + 15
            return {'a': constant_op.constant(37.0), 'b': (constant_op.constant(['Foo']), constant_op.constant('Bar'))}
        tf_value = tf_value_fn()
        iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.from_tensors(tf_value))
        self.assertTrue(structure.are_compatible(dataset_ops.get_structure(iterator), expected_element_structure))
        self.assertEqual(expected_output_classes, dataset_ops.get_legacy_output_classes(iterator))
        self.assertEqual(expected_output_types, dataset_ops.get_legacy_output_types(iterator))
        self.assertEqual(expected_output_shapes, dataset_ops.get_legacy_output_shapes(iterator))

    @combinations.generate(test_base.graph_only_combinations())
    def testIteratorGetNextName(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default():
            iterator = dataset_ops.make_one_shot_iterator(dataset_ops.Dataset.from_tensors(37.0))
            next_element = iterator.get_next(name='overridden_name')
            self.assertEqual('overridden_name', next_element.op.name)

    @combinations.generate(combinations.combine(tf_api_version=[1, 2], mode='eager', execution_mode=[context.ASYNC, context.SYNC]))
    def testIteratorEagerIteration(self, execution_mode):
        if False:
            print('Hello World!')
        with context.eager_mode(), context.execution_mode(execution_mode):
            val = 0
            dataset = dataset_ops.Dataset.range(10)
            iterator = iter(dataset)
            for foo in iterator:
                self.assertEqual(val, foo.numpy())
                val += 1

    @combinations.generate(test_base.eager_only_combinations())
    def testOwnedIteratorFunction(self):
        if False:
            while True:
                i = 10
        queue = data_flow_ops.FIFOQueue(10, dtypes.int64)

        @def_function.function
        def fn():
            if False:
                while True:
                    i = 10
            dataset = dataset_ops.Dataset.range(10)
            iterator = iter(dataset)
            for _ in range(10):
                queue.enqueue(next(iterator))
        fn()
        for i in range(10):
            self.assertEqual(queue.dequeue().numpy(), i)

    @combinations.generate(test_base.eager_only_combinations())
    def testOwnedIteratorFunctionError(self):
        if False:
            return 10
        queue = data_flow_ops.FIFOQueue(10, dtypes.int64)
        queue.enqueue(0)

        def init_fn(n):
            if False:
                print('Hello World!')
            return n

        def next_fn(_):
            if False:
                return 10
            ds = dataset_ops.Dataset.range(0)
            return next(iter(ds))

        def finalize_fn(n):
            if False:
                i = 10
                return i + 15
            queue.enqueue(0)
            return n

        @def_function.function
        def fn():
            if False:
                i = 10
                return i + 15
            output_signature = tensor.TensorSpec((), dtypes.int64)
            dataset = from_generator_op._GeneratorDataset(1, init_fn, next_fn, finalize_fn, output_signature)
            iterator = iter(dataset)
            next(iterator)
        with self.assertRaises(errors.OutOfRangeError):
            fn()
        self.assertEqual(queue.size().numpy(), 2)

    @combinations.generate(test_base.default_test_combinations())
    def testNoInitializer(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(10)
        iterator = iterator_ops.Iterator.from_structure(dataset_ops.get_legacy_output_types(dataset), [])
        with self.assertRaisesRegex(ValueError, 'The iterator does not have an initializer.'):
            _ = iterator.initializer

    @combinations.generate(test_base.default_test_combinations())
    def testtestMissingInput(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, 'When `dataset` is not provided, both `components` and `element_spec` must be specified.'):
            iterator_ops.OwnedIterator(dataset=None)

    @combinations.generate(test_base.eager_only_combinations())
    def testExtraElementSpecInput(self):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.range(1000)
        with self.assertRaisesRegex(ValueError, 'When `dataset` is provided, `element_spec` and `components` must not be specified.'):
            iterator_ops.OwnedIterator(dataset, element_spec=dataset.element_spec)

    @combinations.generate(test_base.eager_only_combinations())
    def testLimitedRetracing(self):
        if False:
            return 10
        trace_count = [0]

        @def_function.function
        def f(iterator):
            if False:
                while True:
                    i = 10
            trace_count[0] += 1
            counter = np.int64(0)
            for elem in iterator:
                counter += elem
            return counter
        dataset = dataset_ops.Dataset.range(5)
        dataset2 = dataset_ops.Dataset.range(10)
        for _ in range(10):
            self.assertEqual(self.evaluate(f(iter(dataset))), 10)
            self.assertEqual(self.evaluate(f(iter(dataset2))), 45)
            self.assertEqual(trace_count[0], 1)

    @combinations.generate(test_base.eager_only_combinations())
    def testNestedFunctionsIteratorResource(self):
        if False:
            i = 10
            return i + 15

        @def_function.function
        def sum_dataset(ds):
            if False:
                print('Hello World!')
            it = iter(ds)

            @def_function.function
            def next_element(it):
                if False:
                    while True:
                        i = 10
                return next(it)
            total = 0
            for _ in range(10):
                total += next_element(it)
            return total
        ds = dataset_ops.Dataset.range(10)
        self.assertEqual(sum_dataset(ds).numpy(), 45)
        self.assertEqual(sum_dataset(ds).numpy(), 45)

    @combinations.generate(test_base.default_test_combinations())
    def testNestedAutomaticControlDependencies(self):
        if False:
            i = 10
            return i + 15
        counter_var = variables.Variable(0)

        def map_fn(x):
            if False:
                i = 10
                return i + 15
            counter_var.assign_add(1)
            return x

        def dataset_fn():
            if False:
                for i in range(10):
                    print('nop')
            return dataset_ops.Dataset.range(10).map(map_fn)

        @def_function.function
        def fn():
            if False:
                for i in range(10):
                    print('nop')
            it = iter(dataset_fn())
            for _ in range(10):
                _ = next(it)
            return counter_var
        self.evaluate(counter_var.initializer)
        self.assertEqual(self.evaluate(fn()), 10)
if __name__ == '__main__':
    test.main()