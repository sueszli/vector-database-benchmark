"""Tests for the currently experimental in-graph batch ops."""
import threading
import time
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import function
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.framework.errors import InvalidArgumentError
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import batch_ops
from tensorflow.python.ops import gen_batch_ops
from tensorflow.python.ops import gen_functional_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.ops import script_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test

def delayed_plus1(x):
    if False:
        while True:
            i = 10
    'Sleeps for 100ms then returns x+1.'
    time.sleep(0.1)
    return x + 1

@test_util.run_all_in_graph_and_eager_modes
class BatchOpsTest(test.TestCase):
    """Tests for batch_ops.{un,}batch."""

    def testBasicBatch(self):
        if False:
            return 10
        'Tests that a single batched tensor executes together and only once.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            (batched, index, _) = batch_ops.batch([inp], num_batch_threads=1, max_batch_size=2, batch_timeout_micros=36000000, grad_timeout_micros=0, batching_queue='')
            thread_results = []

            def worker():
                if False:
                    i = 10
                    return i + 15
                thread_results.extend(sess.run([batched, index], feed_dict={inp: [1]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([batched, index], feed_dict={inp: [2]})
            worker_thread.join()
            if list(thread_results[0][0]):
                batch_t = thread_results[0][0]
                index_t = thread_results[1]
                empty_b = main_results[0][0]
                empty_m = main_results[1]
            else:
                batch_t = main_results[0][0]
                index_t = main_results[1]
                empty_b = thread_results[0][0]
                empty_m = thread_results[1]
            self.assertAllEqual(sorted(batch_t), (1, 2))
            self.assertEqual(len(index_t), 2)
            self.assertEqual(len(empty_b), 0)
            self.assertEqual(len(empty_m), 0)

    def testBatchWithPadding(self):
        if False:
            while True:
                i = 10
        'Test that batching with padding up to an allowed batch size works.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[2])
            (batched, index, _) = batch_ops.batch([inp], num_batch_threads=1, max_batch_size=10, batch_timeout_micros=100000, allowed_batch_sizes=[5, 10], grad_timeout_micros=0, batching_queue='')
            thread_results = []

            def worker():
                if False:
                    for i in range(10):
                        print('nop')
                thread_results.extend(sess.run([batched, index], feed_dict={inp: [1, 3]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([batched, index], feed_dict={inp: [2, 4]})
            worker_thread.join()
            if list(thread_results[0][0]):
                batch_t = thread_results[0][0]
            else:
                batch_t = main_results[0][0]
            self.assertEqual(len(batch_t), 5)

    def testMultipleBatch(self):
        if False:
            print('Hello World!')
        'Tests that multiple batched tensors execute together.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            inp0 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            inp1 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            (batched, _, _) = batch_ops.batch([inp0, inp1], num_batch_threads=1, max_batch_size=2, batch_timeout_micros=36000000, grad_timeout_micros=0, batching_queue='')
            thread_results = []

            def worker():
                if False:
                    print('Hello World!')
                thread_results.extend(sess.run([batched], feed_dict={inp0: [1], inp1: [2]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([batched], feed_dict={inp0: [2], inp1: [3]})
            worker_thread.join()
            if list(thread_results[0][0]):
                batch_t = thread_results[0]
                empty_t = main_results[0]
            else:
                batch_t = main_results[0]
                empty_t = thread_results[0]
            self.assertAllEqual(sorted(batch_t[0]), [1, 2])
            self.assertAllEqual(sorted(batch_t[1]), [2, 3])
            self.assertAllEqual(empty_t[0], [])
            self.assertAllEqual(empty_t[1], [])

    def testIllegalBatchDifferentDim0Sizes(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests illegally feeding tensors with different dim0 sizes.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            inp0 = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            inp1 = array_ops.placeholder(dtype=dtypes.int32, shape=[2])
            (batched, index, _) = batch_ops.batch([inp0, inp1], num_batch_threads=1, max_batch_size=2, batch_timeout_micros=0, grad_timeout_micros=0, batching_queue='')
            with self.assertRaises(Exception) as raised:
                _ = sess.run([batched, index], feed_dict={inp0: [0], inp1: [1, 2]})
            self.assertGreater(raised.exception.message.find('must have equal 0th-dimension size'), 0)

    def testBasicUnbatch(self):
        if False:
            i = 10
            return i + 15
        'Tests that batch and unbatch work together.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            (batched, index, id_t) = batch_ops.batch([inp], num_batch_threads=1, max_batch_size=10, batch_timeout_micros=100000, allowed_batch_sizes=[3, 10], grad_timeout_micros=0, batching_queue='')
            computation = batched[0] + 1
            result = batch_ops.unbatch(computation, index, id_t, timeout_micros=1000000, shared_name='unbatch')
            thread_results = []

            def worker():
                if False:
                    for i in range(10):
                        print('nop')
                thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([result], feed_dict={inp: [2]})
            worker_thread.join()
            self.assertEqual(thread_results[0], [2])
            self.assertEqual(main_results[0], [3])

    def testBasicUnbatchDecorated(self):
        if False:
            return 10
        'Tests that the batch_function decorator works.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            default_inp = array_ops.placeholder_with_default(2, shape=[])

            @batch_ops.batch_function(1, 10, 100000)
            def computation(in_t):
                if False:
                    while True:
                        i = 10
                self.assertTrue(in_t.shape is not None)
                return in_t + 1
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            result = computation(inp)
            thread_results = []

            def worker():
                if False:
                    i = 10
                    return i + 15
                thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([result], feed_dict={inp: [2]})
            worker_thread.join()
            self.assertEqual(thread_results[0], [2])
            self.assertEqual(main_results[0], [3])

    def testUnbatchInvalidIdArg(self):
        if False:
            return 10
        'Tests that unbatch work together.'
        if context.executing_eagerly():
            batched_tensor = constant_op.constant(value=np.random.random(size=(3, 3, 1)), dtype=dtypes.float64)
            batched_index = constant_op.constant(value=np.random.randint(0, 100, size=(3, 3, 1)), dtype=dtypes.int64)
            arg_id = constant_op.constant(value=np.random.randint(0, 100, size=(3, 3, 1)), dtype=dtypes.int64)
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'Input id should be scalar;'):
                batch_ops.unbatch(batched_tensor=batched_tensor, batch_index=batched_index, id=arg_id, timeout_micros=50, container='', shared_name='')

    def testBatchDecoratedWithCapturedInput(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the batch_function decorator works.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            captured_inp0 = array_ops.placeholder_with_default(2.0, shape=[])
            captured_inp1 = resource_variable_ops.ResourceVariable(3.0)
            with ops.device('/cpu:0'):
                captured_inp2 = resource_variable_ops.ResourceVariable(4.0)

            @batch_ops.batch_function(1, 10, 100000)
            def computation(in_t):
                if False:
                    for i in range(10):
                        print('nop')
                return in_t + captured_inp0 + captured_inp1 + captured_inp2
            inp = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
            result = computation(inp)
            thread_results = []

            def worker():
                if False:
                    for i in range(10):
                        print('nop')
                thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
            sess.run(variables.global_variables_initializer())
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([result], feed_dict={inp: [2]})
            worker_thread.join()
            self.assertEqual(thread_results[0], [10])
            self.assertEqual(main_results[0], [11])

    @test_util.disable_xla('DeviceIndex returns sentinel value with XLA')
    def testBatchDecoratedGpu(self):
        if False:
            return 10
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:

            @batch_ops.batch_function(1, 10, 100000)
            def computation(in_t):
                if False:
                    while True:
                        i = 10
                index = gen_functional_ops.DeviceIndex(device_names=['CPU', 'GPU'])
                return in_t + math_ops.cast(index, dtypes.float32)
            inp = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
            result = computation(inp)
            thread_results = []

            def worker():
                if False:
                    print('Hello World!')
                thread_results.extend(sess.run([result], feed_dict={inp: [10.0]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([result], feed_dict={inp: [20.0]})
            worker_thread.join()
            self.assertEqual(thread_results[0], [10 + test_util.is_gpu_available()])
            self.assertEqual(main_results[0], [20 + test_util.is_gpu_available()])

    def testParallelRunsWithCpuAndGpu(self):
        if False:
            return 10
        if context.executing_eagerly():
            return

        @batch_ops.batch_function(1, 2, 1)
        def f(x):
            if False:
                while True:
                    i = 10
            with ops.device('/GPU:0'):
                x = x + 1.0
            with ops.device('/CPU:0'):
                return x + 1
        num_calls = 10
        placeholders = [array_ops.placeholder(dtypes.float32, shape=(1,)) for _ in range(num_calls)]
        results = []
        for p in placeholders:
            result = f(p)
            results.append(result)
        inputs = [[float(i)] for i in range(num_calls)]
        expected = [[float(i + 2)] for i in range(num_calls)]
        with self.session() as sess:
            outputs = sess.run(results, feed_dict=dict(zip(placeholders, inputs)))
            self.assertAllEqual(outputs, expected)

    def testSoftPlacement(self):
        if False:
            return 10
        if context.executing_eagerly():
            return

        @batch_ops.batch_function(1, 10, 100000)
        def computation(in_t):
            if False:
                i = 10
                return i + 15
            with ops.device('/GPU:0'):
                return in_t + 1.0
        inp = array_ops.placeholder(dtype=dtypes.float32, shape=[1])
        result = computation(inp)
        config = config_pb2.ConfigProto(allow_soft_placement=True)
        with self.session(config=config) as sess:
            sess.run([result], feed_dict={inp: [20.0]})
        config.allow_soft_placement = False
        with self.session(config=config) as sess:
            if test_util.is_gpu_available():
                sess.run([result], feed_dict={inp: [20.0]})
            else:
                with self.assertRaisesRegex(InvalidArgumentError, 'Cannot assign a device for operation'):
                    sess.run([result], feed_dict={inp: [20.0]})

    def testBatchFunctionOp(self):
        if False:
            for i in range(10):
                print('nop')
        'Tests that the batch_function op works.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:

            @function.Defun(dtypes.int32)
            def computation(in_t):
                if False:
                    return 10
                return in_t + 1
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            result = gen_batch_ops.batch_function([inp], num_batch_threads=1, max_batch_size=10, batch_timeout_micros=100000, Tout=[dtypes.int32], f=computation, captured_tensors=computation.captured_inputs)
            thread_results = []

            def worker():
                if False:
                    while True:
                        i = 10
                thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([result], feed_dict={inp: [2]})
            worker_thread.join()
            self.assertEqual(thread_results[0], [2])
            self.assertEqual(main_results[0], [3])

    def testBatchFunctionOpWithCapturedInput(self):
        if False:
            while True:
                i = 10
        'Tests that batch_function op works with captured input.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            captured_inp0 = array_ops.placeholder_with_default(2, shape=[])
            captured_inp1 = array_ops.placeholder_with_default(1, shape=[])
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])

            @function.Defun(dtypes.int32)
            def computation(inp):
                if False:
                    print('Hello World!')
                return inp + captured_inp0 - captured_inp1
            result = gen_batch_ops.batch_function(num_batch_threads=1, max_batch_size=10, batch_timeout_micros=100000, allowed_batch_sizes=[3, 10], batching_queue='', f=computation, in_tensors=[inp], captured_tensors=computation.captured_inputs, Tout=[o.type for o in computation.definition.signature.output_arg])
            thread_results = []

            def worker():
                if False:
                    while True:
                        i = 10
                thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([result], feed_dict={inp: [2]})
            worker_thread.join()
            self.assertEqual(thread_results[0], [2])
            self.assertEqual(main_results[0], [3])

    def testBatchFunctionOpWithInputError(self):
        if False:
            while True:
                i = 10
        'Tests that batch_function op works with error in the inputs.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])

            @function.Defun(dtypes.int32, dtypes.int32)
            def computation(in0, in1):
                if False:
                    for i in range(10):
                        print('nop')
                return in0 + in1
            result = gen_batch_ops.batch_function([inp], num_batch_threads=1, max_batch_size=10, batch_timeout_micros=100000, batching_queue='', f=computation, captured_tensors=computation.captured_inputs, Tout=[o.type for o in computation.definition.signature.output_arg])
            with self.assertRaisesRegex(InvalidArgumentError, 'Function takes 2 argument\\(s\\) but 1 argument\\(s\\) were passed'):
                sess.run([result], feed_dict={inp: [2]})

    def testBatchFunctionOpWithLargeBatchSplitted(self):
        if False:
            return 10
        'Tests that the batch_function op works with large batch splitted.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:

            @function.Defun(dtypes.int32)
            def computation(in_t):
                if False:
                    for i in range(10):
                        print('nop')
                return in_t + 3
            inp = array_ops.placeholder(dtype=dtypes.int32)
            result = gen_batch_ops.batch_function([inp], num_batch_threads=2, allowed_batch_sizes=[1, 2], max_batch_size=5, batch_timeout_micros=100000, Tout=[dtypes.int32], enable_large_batch_splitting=True, f=computation, captured_tensors=computation.captured_inputs)
            thread1_results = []
            thread2_results = []

            def worker1():
                if False:
                    for i in range(10):
                        print('nop')
                thread1_results.extend(sess.run([result], feed_dict={inp: [5, 6, 7, 8, 9]}))
            worker_thread1 = threading.Thread(target=worker1)
            worker_thread1.start()

            def worker2():
                if False:
                    i = 10
                    return i + 15
                thread2_results.extend(sess.run([result], feed_dict={inp: [10]}))
            worker_thread2 = threading.Thread(target=worker2)
            worker_thread2.start()
            main_results = sess.run([result], feed_dict={inp: [2, 3, 4]})
            worker_thread1.join()
            worker_thread2.join()
            self.assertTrue(np.all(np.equal(thread2_results[0], np.array([13], dtype=np.int32))))
            self.assertTrue(np.all(np.equal(thread1_results[0], np.array([8, 9, 10, 11, 12], dtype=np.int32))))
            self.assertTrue(np.all(np.equal(main_results[0], np.array([5, 6, 7], dtype=np.int32))))

    def testBasicUnbatchDecoratedWithReshape(self):
        if False:
            while True:
                i = 10
        'Tests that the batch_function decorator works.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:

            @batch_ops.batch_function(1, 10, 100000)
            def computation(in_t):
                if False:
                    for i in range(10):
                        print('nop')
                return array_ops.reshape(in_t, [-1]) + 1
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1, 1])
            result = computation(inp)
            thread_results = []

            def worker():
                if False:
                    i = 10
                    return i + 15
                thread_results.extend(sess.run([result], feed_dict={inp: [[1]]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            main_results = sess.run([result], feed_dict={inp: [[2]]})
            worker_thread.join()
            self.assertEqual(thread_results[0], [2])
            self.assertEqual(main_results[0], [3])

    def testUnbatchTimeout(self):
        if False:
            i = 10
            return i + 15
        'Tests that the unbatch timeout works.'
        if context.executing_eagerly():
            return
        with self.cached_session() as sess:
            inp = array_ops.placeholder(dtype=dtypes.int32, shape=[1])
            (batched, index, id_t) = batch_ops.batch([inp], num_batch_threads=1, max_batch_size=2, batch_timeout_micros=36000000, grad_timeout_micros=0, batching_queue='')
            computation = batched[0] + 1
            timeout_micros = 10
            result = batch_ops.unbatch(computation, index, id_t, timeout_micros, shared_name='shared_unbatch')
            computation_delayed = script_ops.py_func(delayed_plus1, [batched[0]], dtypes.int32)
            result_delayed = batch_ops.unbatch(computation_delayed, index, id_t, timeout_micros, shared_name='shared_unbatch')
            thread_results = []

            def worker():
                if False:
                    while True:
                        i = 10
                thread_results.extend(sess.run([result], feed_dict={inp: [1]}))
            worker_thread = threading.Thread(target=worker)
            worker_thread.start()
            time.sleep(0.1)
            _ = sess.run([result_delayed], feed_dict={inp: [2]})
            worker_thread.join()
            self.assertEqual(len(thread_results), 0)

    def testUnbatchGradInvalidId(self):
        if False:
            print('Hello World!')
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(gen_batch_ops.unbatch_grad(original_input=constant_op.constant([1]), batch_index=constant_op.constant([[0, 0, 0]], dtype=dtypes.int64), grad=constant_op.constant([1]), id=constant_op.constant([1, 1], dtype=dtypes.int64)))

    def testUnbatchGradInvalidBatchId(self):
        if False:
            i = 10
            return i + 15
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(gen_batch_ops.unbatch_grad(original_input=constant_op.constant([1]), batch_index=constant_op.constant([[0, 0]], dtype=dtypes.int64), grad=constant_op.constant([1]), id=constant_op.constant([1], dtype=dtypes.int64)))

    def testUnbatchGradInvalidArgs(self):
        if False:
            while True:
                i = 10
        original_input = random_ops.random_uniform(shape=(3, 1), dtype=dtypes.float64, maxval=None)
        batch_index = random_ops.random_uniform(shape=(3, 1), dtype=dtypes.int64, maxval=65536)
        grad = random_ops.random_uniform(shape=(3, 1), dtype=dtypes.float64, maxval=None)
        batch_id = random_ops.random_uniform(shape=(3, 1), dtype=dtypes.int64, maxval=65536)
        with self.assertRaises(errors.InvalidArgumentError):
            self.evaluate(gen_batch_ops.unbatch_grad(original_input=original_input, batch_index=batch_index, grad=grad, id=batch_id, container='', shared_name='', name=''))
if __name__ == '__main__':
    test.main()