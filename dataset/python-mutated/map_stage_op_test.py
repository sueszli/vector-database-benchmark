import queue
import threading
import numpy as np
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
TIMEOUT = 1

class MapStageTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testSimple(self):
        if False:
            return 10
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                pi = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32])
                stage = stager.put(pi, [v], [0])
                (k, y) = stager.get(gi)
                y = math_ops.reduce_max(math_ops.matmul(y, y))
        g.finalize()
        with self.session(graph=g) as sess:
            sess.run(stage, feed_dict={x: -1, pi: 0})
            for i in range(10):
                (_, yval) = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
                self.assertAllClose(4 * (i - 1) * (i - 1) * 128, yval, rtol=0.0001)

    @test_util.run_deprecated_v1
    def testMultiple(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                pi = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32])
                stage = stager.put(pi, [x, v], [0, 1])
                (k, (z, y)) = stager.get(gi)
                y = math_ops.reduce_max(z * math_ops.matmul(y, y))
        g.finalize()
        with self.session(graph=g) as sess:
            sess.run(stage, feed_dict={x: -1, pi: 0})
            for i in range(10):
                (_, yval) = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
                self.assertAllClose(4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=0.0001)

    @test_util.run_deprecated_v1
    def testDictionary(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                pi = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32], shapes=[[], [128, 128]], names=['x', 'v'])
                stage = stager.put(pi, {'x': x, 'v': v})
                (key, ret) = stager.get(gi)
                z = ret['x']
                y = ret['v']
                y = math_ops.reduce_max(z * math_ops.matmul(y, y))
        g.finalize()
        with self.session(graph=g) as sess:
            sess.run(stage, feed_dict={x: -1, pi: 0})
            for i in range(10):
                (_, yval) = sess.run([stage, y], feed_dict={x: i, pi: i + 1, gi: i})
                self.assertAllClose(4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=0.0001)

    def testColocation(self):
        if False:
            while True:
                i = 10
        gpu_dev = test.gpu_device_name()
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(gpu_dev):
                stager = data_flow_ops.MapStagingArea([dtypes.float32])
                y = stager.put(1, [v], [0])
                expected_name = gpu_dev if 'gpu' not in gpu_dev else '/device:GPU:0'
                self.assertEqual(y.device, expected_name)
            with ops.device('/cpu:0'):
                (_, x) = stager.get(1)
                y = stager.peek(1)[0]
                (_, z) = stager.get()
                self.assertEqual(x[0].device, '/device:CPU:0')
                self.assertEqual(y.device, '/device:CPU:0')
                self.assertEqual(z[0].device, '/device:CPU:0')
        g.finalize()

    @test_util.run_deprecated_v1
    def testPeek(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.int32, name='x')
                pi = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
                p = array_ops.placeholder(dtypes.int32, name='p')
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.int32], shapes=[[]])
                stage = stager.put(pi, [x], [0])
                peek = stager.peek(gi)
                size = stager.size()
        g.finalize()
        n = 10
        with self.session(graph=g) as sess:
            for i in range(n):
                sess.run(stage, feed_dict={x: i, pi: i})
            for i in range(n):
                self.assertEqual(sess.run(peek, feed_dict={gi: i})[0], i)
            self.assertEqual(sess.run(size), 10)

    @test_util.run_deprecated_v1
    def testSizeAndClear(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32, name='x')
                pi = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32], shapes=[[], [128, 128]], names=['x', 'v'])
                stage = stager.put(pi, {'x': x, 'v': v})
                size = stager.size()
                clear = stager.clear()
        g.finalize()
        with self.session(graph=g) as sess:
            sess.run(stage, feed_dict={x: -1, pi: 3})
            self.assertEqual(sess.run(size), 1)
            sess.run(stage, feed_dict={x: -1, pi: 1})
            self.assertEqual(sess.run(size), 2)
            sess.run(clear)
            self.assertEqual(sess.run(size), 0)

    @test_util.run_deprecated_v1
    def testCapacity(self):
        if False:
            for i in range(10):
                print('nop')
        capacity = 3
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.int32, name='x')
                pi = array_ops.placeholder(dtypes.int64, name='pi')
                gi = array_ops.placeholder(dtypes.int64, name='gi')
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.int32], capacity=capacity, shapes=[[]])
            stage = stager.put(pi, [x], [0])
            get = stager.get()
            size = stager.size()
        g.finalize()
        value_queue = queue.Queue()
        n = 8
        with self.session(graph=g) as sess:

            def thread_run():
                if False:
                    i = 10
                    return i + 15
                for i in range(n):
                    sess.run(stage, feed_dict={x: i, pi: i})
                    value_queue.put(0)
            t = threading.Thread(target=thread_run)
            t.daemon = True
            t.start()
            try:
                for i in range(n):
                    value_queue.get(timeout=TIMEOUT)
            except queue.Empty:
                pass
            if not i == capacity:
                self.fail("Expected to timeout on iteration '{}' but instead timed out on iteration '{}' Staging Area size is '{}' and configured capacity is '{}'.".format(capacity, i, sess.run(size), capacity))
            self.assertEqual(sess.run(size), capacity)
            for i in range(n):
                sess.run(get)
            self.assertEqual(sess.run(size), 0)

    @test_util.run_deprecated_v1
    def testMemoryLimit(self):
        if False:
            return 10
        memory_limit = 512 * 1024
        chunk = 200 * 1024
        capacity = memory_limit // chunk
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.uint8, name='x')
                pi = array_ops.placeholder(dtypes.int64, name='pi')
                gi = array_ops.placeholder(dtypes.int64, name='gi')
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.uint8], memory_limit=memory_limit, shapes=[[]])
                stage = stager.put(pi, [x], [0])
                get = stager.get()
                size = stager.size()
        g.finalize()
        value_queue = queue.Queue()
        n = 8
        with self.session(graph=g) as sess:

            def thread_run():
                if False:
                    while True:
                        i = 10
                for i in range(n):
                    data = np.full(chunk, i, dtype=np.uint8)
                    sess.run(stage, feed_dict={x: data, pi: i})
                    value_queue.put(0)
            t = threading.Thread(target=thread_run)
            t.daemon = True
            t.start()
            try:
                for i in range(n):
                    value_queue.get(timeout=TIMEOUT)
            except queue.Empty:
                pass
            if not i == capacity:
                self.fail("Expected to timeout on iteration '{}' but instead timed out on iteration '{}' Staging Area size is '{}' and configured capacity is '{}'.".format(capacity, i, sess.run(size), capacity))
            self.assertEqual(sess.run(size), capacity)
            for i in range(n):
                sess.run(get)
            self.assertEqual(sess.run(size), 0)

    @test_util.run_deprecated_v1
    def testOrdering(self):
        if False:
            while True:
                i = 10
        import random
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.int32, name='x')
                pi = array_ops.placeholder(dtypes.int64, name='pi')
                gi = array_ops.placeholder(dtypes.int64, name='gi')
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.int32], shapes=[[]], ordered=True)
                stage = stager.put(pi, [x], [0])
                get = stager.get()
                size = stager.size()
        g.finalize()
        n = 10
        with self.session(graph=g) as sess:
            keys = list(reversed(range(n)))
            for i in keys:
                sess.run(stage, feed_dict={pi: i, x: i})
            self.assertEqual(sess.run(size), n)
            for (i, k) in enumerate(reversed(keys)):
                (get_key, values) = sess.run(get)
                self.assertTrue(i == k == get_key == values)
            self.assertEqual(sess.run(size), 0)

    @test_util.run_deprecated_v1
    def testPartialDictInsert(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                f = array_ops.placeholder(dtypes.float32)
                v = array_ops.placeholder(dtypes.float32)
                pi = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32, dtypes.float32], names=['x', 'v', 'f'])
                stage_xf = stager.put(pi, {'x': x, 'f': f})
                stage_v = stager.put(pi, {'v': v})
                (key, ret) = stager.get(gi)
                size = stager.size()
                isize = stager.incomplete_size()
        g.finalize()
        with self.session(graph=g) as sess:
            self.assertEqual(sess.run([size, isize]), [0, 0])
            sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
            self.assertEqual(sess.run([size, isize]), [0, 1])
            sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
            self.assertEqual(sess.run([size, isize]), [0, 2])
            sess.run(stage_v, feed_dict={pi: 0, v: 1})
            self.assertEqual(sess.run([size, isize]), [1, 1])
            self.assertEqual(sess.run([key, ret], feed_dict={gi: 0}), [0, {'x': 1, 'f': 2, 'v': 1}])
            self.assertEqual(sess.run([size, isize]), [0, 1])
            sess.run(stage_v, feed_dict={pi: 1, v: 3})
            self.assertEqual(sess.run([key, ret], feed_dict={gi: 1}), [1, {'x': 1, 'f': 2, 'v': 3}])

    @test_util.run_deprecated_v1
    def testPartialIndexInsert(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                f = array_ops.placeholder(dtypes.float32)
                v = array_ops.placeholder(dtypes.float32)
                pi = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32, dtypes.float32])
                stage_xf = stager.put(pi, [x, f], [0, 2])
                stage_v = stager.put(pi, [v], [1])
                (key, ret) = stager.get(gi)
                size = stager.size()
                isize = stager.incomplete_size()
        g.finalize()
        with self.session(graph=g) as sess:
            self.assertEqual(sess.run([size, isize]), [0, 0])
            sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
            self.assertEqual(sess.run([size, isize]), [0, 1])
            sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
            self.assertEqual(sess.run([size, isize]), [0, 2])
            sess.run(stage_v, feed_dict={pi: 0, v: 1})
            self.assertEqual(sess.run([size, isize]), [1, 1])
            self.assertEqual(sess.run([key, ret], feed_dict={gi: 0}), [0, [1, 1, 2]])
            self.assertEqual(sess.run([size, isize]), [0, 1])
            sess.run(stage_v, feed_dict={pi: 1, v: 3})
            self.assertEqual(sess.run([key, ret], feed_dict={gi: 1}), [1, [1, 3, 2]])

    @test_util.run_deprecated_v1
    def testPartialDictGetsAndPeeks(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                f = array_ops.placeholder(dtypes.float32)
                v = array_ops.placeholder(dtypes.float32)
                pi = array_ops.placeholder(dtypes.int64)
                pei = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32, dtypes.float32], names=['x', 'v', 'f'])
                stage_xf = stager.put(pi, {'x': x, 'f': f})
                stage_v = stager.put(pi, {'v': v})
                peek_xf = stager.peek(pei, ['x', 'f'])
                peek_v = stager.peek(pei, ['v'])
                (key_xf, get_xf) = stager.get(gi, ['x', 'f'])
                (key_v, get_v) = stager.get(gi, ['v'])
                (pop_key_xf, pop_xf) = stager.get(indices=['x', 'f'])
                (pop_key_v, pop_v) = stager.get(pi, ['v'])
                size = stager.size()
                isize = stager.incomplete_size()
        g.finalize()
        with self.session(graph=g) as sess:
            self.assertEqual(sess.run([size, isize]), [0, 0])
            sess.run(stage_xf, feed_dict={pi: 0, x: 1, f: 2})
            self.assertEqual(sess.run([size, isize]), [0, 1])
            sess.run(stage_xf, feed_dict={pi: 1, x: 1, f: 2})
            self.assertEqual(sess.run([size, isize]), [0, 2])
            sess.run(stage_v, feed_dict={pi: 0, v: 1})
            self.assertEqual(sess.run([size, isize]), [1, 1])
            self.assertEqual(sess.run(peek_xf, feed_dict={pei: 0}), {'x': 1, 'f': 2})
            self.assertEqual(sess.run(peek_v, feed_dict={pei: 0}), {'v': 1})
            self.assertEqual(sess.run([size, isize]), [1, 1])
            self.assertEqual(sess.run([key_xf, get_xf], feed_dict={gi: 0}), [0, {'x': 1, 'f': 2}])
            self.assertEqual(sess.run([size, isize]), [1, 1])
            with self.assertRaises(errors.InvalidArgumentError) as cm:
                sess.run([key_xf, get_xf], feed_dict={gi: 0})
            exc_str = "Tensor at index '0' for key '0' has already been removed."
            self.assertIn(exc_str, cm.exception.message)
            self.assertEqual(sess.run([key_v, get_v], feed_dict={gi: 0}), [0, {'v': 1}])
            self.assertEqual(sess.run([size, isize]), [0, 1])
            sess.run(stage_v, feed_dict={pi: 1, v: 1})
            self.assertEqual(sess.run([size, isize]), [1, 0])
            self.assertEqual(sess.run([pop_key_xf, pop_xf]), [1, {'x': 1, 'f': 2}])
            self.assertEqual(sess.run([size, isize]), [1, 0])
            self.assertEqual(sess.run([pop_key_v, pop_v], feed_dict={pi: 1}), [1, {'v': 1}])
            self.assertEqual(sess.run([size, isize]), [0, 0])

    @test_util.run_deprecated_v1
    def testPartialIndexGets(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default() as g:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                f = array_ops.placeholder(dtypes.float32)
                v = array_ops.placeholder(dtypes.float32)
                pi = array_ops.placeholder(dtypes.int64)
                pei = array_ops.placeholder(dtypes.int64)
                gi = array_ops.placeholder(dtypes.int64)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.MapStagingArea([dtypes.float32, dtypes.float32, dtypes.float32])
                stage_xvf = stager.put(pi, [x, v, f], [0, 1, 2])
                (key_xf, get_xf) = stager.get(gi, [0, 2])
                (key_v, get_v) = stager.get(gi, [1])
                size = stager.size()
                isize = stager.incomplete_size()
        g.finalize()
        with self.session(graph=g) as sess:
            sess.run(stage_xvf, feed_dict={pi: 0, x: 1, f: 2, v: 3})
            self.assertEqual(sess.run([size, isize]), [1, 0])
            self.assertEqual(sess.run([key_xf, get_xf], feed_dict={gi: 0}), [0, [1, 2]])
            self.assertEqual(sess.run([size, isize]), [1, 0])
            self.assertEqual(sess.run([key_v, get_v], feed_dict={gi: 0}), [0, [3]])
            self.assertEqual(sess.run([size, isize]), [0, 0])

    @test_util.run_deprecated_v1
    def testNonScalarKeyOrderedMap(self):
        if False:
            for i in range(10):
                print('nop')
        with ops.Graph().as_default() as g:
            x = array_ops.placeholder(dtypes.float32)
            v = 2.0 * (array_ops.zeros([128, 128]) + x)
            t = data_flow_ops.gen_data_flow_ops.ordered_map_stage(key=constant_op.constant(value=[1], shape=(1, 3), dtype=dtypes.int64), indices=np.array([[6]]), values=[x, v], dtypes=[dtypes.int64], capacity=0, memory_limit=0, container='container1', shared_name='', name=None)
        g.finalize()
        with self.session(graph=g) as sess:
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'key must be an int64 scalar'):
                sess.run(t, feed_dict={x: 1})

    @test_util.run_deprecated_v1
    def testNonScalarKeyUnorderedMap(self):
        if False:
            print('Hello World!')
        with ops.Graph().as_default() as g:
            x = array_ops.placeholder(dtypes.float32)
            v = 2.0 * (array_ops.zeros([128, 128]) + x)
            t = data_flow_ops.gen_data_flow_ops.map_stage(key=constant_op.constant(value=[1], shape=(1, 3), dtype=dtypes.int64), indices=np.array([[6]]), values=[x, v], dtypes=[dtypes.int64], capacity=0, memory_limit=0, container='container1', shared_name='', name=None)
        g.finalize()
        with self.session(graph=g) as sess:
            with self.assertRaisesRegex(errors.InvalidArgumentError, 'key must be an int64 scalar'):
                sess.run(t, feed_dict={x: 1})

    def testNonScalarKeyMapPeek(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(errors.InvalidArgumentError, 'key must be an int64 scalar'):
            v = data_flow_ops.gen_data_flow_ops.map_peek(key=constant_op.constant(value=[1], shape=(1, 3), dtype=dtypes.int64), indices=np.array([[6]]), dtypes=[dtypes.int64], capacity=0, memory_limit=0, container='container1', shared_name='', name=None)
            self.evaluate(v)
if __name__ == '__main__':
    test.main()