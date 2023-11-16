import queue
import threading
import numpy as np
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test
TIMEOUT = 1

class StageTest(test.TestCase):

    @test_util.run_deprecated_v1
    def testSimple(self):
        if False:
            return 10
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.StagingArea([dtypes.float32])
                stage = stager.put([v])
                y = stager.get()
                y = math_ops.reduce_max(math_ops.matmul(y, y))
        G.finalize()
        with self.session(graph=G) as sess:
            sess.run(stage, feed_dict={x: -1})
            for i in range(10):
                (_, yval) = sess.run([stage, y], feed_dict={x: i})
                self.assertAllClose(4 * (i - 1) * (i - 1) * 128, yval, rtol=0.0001)

    @test_util.run_deprecated_v1
    def testMultiple(self):
        if False:
            while True:
                i = 10
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.StagingArea([dtypes.float32, dtypes.float32])
                stage = stager.put([x, v])
                (z, y) = stager.get()
                y = math_ops.reduce_max(z * math_ops.matmul(y, y))
        G.finalize()
        with self.session(graph=G) as sess:
            sess.run(stage, feed_dict={x: -1})
            for i in range(10):
                (_, yval) = sess.run([stage, y], feed_dict={x: i})
                self.assertAllClose(4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=0.0001)

    @test_util.run_deprecated_v1
    def testDictionary(self):
        if False:
            return 10
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.StagingArea([dtypes.float32, dtypes.float32], shapes=[[], [128, 128]], names=['x', 'v'])
                stage = stager.put({'x': x, 'v': v})
                ret = stager.get()
                z = ret['x']
                y = ret['v']
                y = math_ops.reduce_max(z * math_ops.matmul(y, y))
        G.finalize()
        with self.session(graph=G) as sess:
            sess.run(stage, feed_dict={x: -1})
            for i in range(10):
                (_, yval) = sess.run([stage, y], feed_dict={x: i})
                self.assertAllClose(4 * (i - 1) * (i - 1) * (i - 1) * 128, yval, rtol=0.0001)

    def testColocation(self):
        if False:
            print('Hello World!')
        gpu_dev = test.gpu_device_name()
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32)
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(gpu_dev):
                stager = data_flow_ops.StagingArea([dtypes.float32])
                y = stager.put([v])
                expected_name = gpu_dev if 'gpu' not in gpu_dev else '/device:GPU:0'
                self.assertEqual(y.device, expected_name)
            with ops.device('/cpu:0'):
                x = stager.get()[0]
                self.assertEqual(x.device, '/device:CPU:0')
        G.finalize()

    @test_util.run_deprecated_v1
    def testPeek(self):
        if False:
            i = 10
            return i + 15
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.int32, name='x')
                p = array_ops.placeholder(dtypes.int32, name='p')
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.StagingArea([dtypes.int32], shapes=[[]])
                stage = stager.put([x])
                peek = stager.peek(p)
                ret = stager.get()
        G.finalize()
        with self.session(graph=G) as sess:
            for i in range(10):
                sess.run(stage, feed_dict={x: i})
            for i in range(10):
                self.assertTrue(sess.run(peek, feed_dict={p: i}) == [i])

    def testPeekBadIndex(self):
        if False:
            i = 10
            return i + 15
        stager = data_flow_ops.StagingArea([dtypes.int32], shapes=[[10]])
        stager.put([array_ops.zeros([10], dtype=dtypes.int32)])
        with self.assertRaisesRegex((ValueError, errors.InvalidArgumentError), 'must be scalar'):
            self.evaluate(stager.peek([]))

    @test_util.run_deprecated_v1
    def testSizeAndClear(self):
        if False:
            return 10
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.float32, name='x')
                v = 2.0 * (array_ops.zeros([128, 128]) + x)
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.StagingArea([dtypes.float32, dtypes.float32], shapes=[[], [128, 128]], names=['x', 'v'])
                stage = stager.put({'x': x, 'v': v})
                ret = stager.get()
                size = stager.size()
                clear = stager.clear()
        G.finalize()
        with self.session(graph=G) as sess:
            sess.run(stage, feed_dict={x: -1})
            self.assertEqual(sess.run(size), 1)
            sess.run(stage, feed_dict={x: -1})
            self.assertEqual(sess.run(size), 2)
            sess.run(clear)
            self.assertEqual(sess.run(size), 0)

    @test_util.run_deprecated_v1
    def testCapacity(self):
        if False:
            print('Hello World!')
        self.skipTest('b/123423516 this test is flaky on gpu.')
        capacity = 3
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.int32, name='x')
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.StagingArea([dtypes.int32], capacity=capacity, shapes=[[]])
                stage = stager.put([x])
                ret = stager.get()
                size = stager.size()
        G.finalize()
        value_queue = queue.Queue()
        n = 8
        with self.session(graph=G) as sess:

            def thread_run():
                if False:
                    print('Hello World!')
                for i in range(n):
                    sess.run(stage, feed_dict={x: i})
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
            self.assertTrue(sess.run(size) == capacity)
            for i in range(n):
                self.assertTrue(sess.run(ret) == [i])
            self.assertTrue(sess.run(size) == 0)

    @test_util.run_deprecated_v1
    def testMemoryLimit(self):
        if False:
            return 10
        memory_limit = 512 * 1024
        chunk = 200 * 1024
        capacity = memory_limit // chunk
        with ops.Graph().as_default() as G:
            with ops.device('/cpu:0'):
                x = array_ops.placeholder(dtypes.uint8, name='x')
            with ops.device(test.gpu_device_name()):
                stager = data_flow_ops.StagingArea([dtypes.uint8], memory_limit=memory_limit, shapes=[[]])
                stage = stager.put([x])
                ret = stager.get()
                size = stager.size()
        G.finalize()
        value_queue = queue.Queue()
        n = 8
        with self.session(graph=G) as sess:

            def thread_run():
                if False:
                    for i in range(10):
                        print('nop')
                for i in range(n):
                    sess.run(stage, feed_dict={x: np.full(chunk, i, dtype=np.uint8)})
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
            self.assertTrue(sess.run(size) == capacity)
            for i in range(n):
                self.assertTrue(np.all(sess.run(ret)[0] == i))
            self.assertTrue(sess.run(size) == 0)
if __name__ == '__main__':
    test.main()