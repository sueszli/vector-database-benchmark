"""Tests for tensorflow.ops.data_flow_ops.FIFOQueue."""
import time
from tensorflow.compiler.tests import xla_test
from tensorflow.python.framework import dtypes as dtypes_lib
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.platform import test

class FIFOQueueTest(xla_test.XLATestCase):

    def testEnqueue(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
            enqueue_op = q.enqueue((10.0,))
            enqueue_op.run()

    def testEnqueueWithShape(self):
        if False:
            while True:
                i = 10
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32, shapes=(3, 2))
            enqueue_correct_op = q.enqueue(([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],))
            enqueue_correct_op.run()
            with self.assertRaises(ValueError):
                q.enqueue(([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],))
            self.assertEqual(1, self.evaluate(q.size()))

    def testMultipleDequeues(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
            self.evaluate(q.enqueue([1]))
            self.evaluate(q.enqueue([2]))
            self.evaluate(q.enqueue([3]))
            (a, b, c) = self.evaluate([q.dequeue(), q.dequeue(), q.dequeue()])
            self.assertAllEqual(set([1, 2, 3]), set([a, b, c]))

    def testQueuesDontShare(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
            self.evaluate(q.enqueue(1))
            q2 = data_flow_ops.FIFOQueue(10, [dtypes_lib.int32], shapes=[()])
            self.evaluate(q2.enqueue(2))
            self.assertAllEqual(self.evaluate(q2.dequeue()), 2)
            self.assertAllEqual(self.evaluate(q.dequeue()), 1)

    def testEnqueueDictWithoutNames(self):
        if False:
            return 10
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
            with self.assertRaisesRegex(ValueError, 'must have names'):
                q.enqueue({'a': 12.0})

    def testParallelEnqueue(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session() as sess, self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
            elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            enqueue_ops = [q.enqueue((x,)) for x in elems]
            dequeued_t = q.dequeue()

            def enqueue(enqueue_op):
                if False:
                    for i in range(10):
                        print('nop')
                sess.run(enqueue_op)
            threads = [self.checkedThread(target=enqueue, args=(e,)) for e in enqueue_ops]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            results = []
            for _ in range(len(elems)):
                results.append(self.evaluate(dequeued_t))
            self.assertItemsEqual(elems, results)

    def testParallelDequeue(self):
        if False:
            return 10
        with self.session() as sess, self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
            elems = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0]
            enqueue_ops = [q.enqueue((x,)) for x in elems]
            dequeued_t = q.dequeue()
            for enqueue_op in enqueue_ops:
                enqueue_op.run()
            results = []

            def dequeue():
                if False:
                    i = 10
                    return i + 15
                results.append(sess.run(dequeued_t))
            threads = [self.checkedThread(target=dequeue) for _ in enqueue_ops]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertItemsEqual(elems, results)

    def testDequeue(self):
        if False:
            for i in range(10):
                print('nop')
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
            elems = [10.0, 20.0, 30.0]
            enqueue_ops = [q.enqueue((x,)) for x in elems]
            dequeued_t = q.dequeue()
            for enqueue_op in enqueue_ops:
                enqueue_op.run()
            for i in range(len(elems)):
                vals = self.evaluate(dequeued_t)
                self.assertEqual([elems[i]], vals)

    def testEnqueueAndBlockingDequeue(self):
        if False:
            i = 10
            return i + 15
        with self.session() as sess, self.test_scope():
            q = data_flow_ops.FIFOQueue(3, dtypes_lib.float32)
            elems = [10.0, 20.0, 30.0]
            enqueue_ops = [q.enqueue((x,)) for x in elems]
            dequeued_t = q.dequeue()

            def enqueue():
                if False:
                    while True:
                        i = 10
                time.sleep(0.1)
                for enqueue_op in enqueue_ops:
                    sess.run(enqueue_op)
            results = []

            def dequeue():
                if False:
                    while True:
                        i = 10
                for _ in range(len(elems)):
                    results.append(sess.run(dequeued_t))
            enqueue_thread = self.checkedThread(target=enqueue)
            dequeue_thread = self.checkedThread(target=dequeue)
            enqueue_thread.start()
            dequeue_thread.start()
            enqueue_thread.join()
            dequeue_thread.join()
            for (elem, result) in zip(elems, results):
                self.assertEqual([elem], result)

    def testMultiEnqueueAndDequeue(self):
        if False:
            return 10
        with self.session() as sess, self.test_scope():
            q = data_flow_ops.FIFOQueue(10, (dtypes_lib.int32, dtypes_lib.float32))
            elems = [(5, 10.0), (10, 20.0), (15, 30.0)]
            enqueue_ops = [q.enqueue((x, y)) for (x, y) in elems]
            dequeued_t = q.dequeue()
            for enqueue_op in enqueue_ops:
                enqueue_op.run()
            for i in range(len(elems)):
                (x_val, y_val) = sess.run(dequeued_t)
                (x, y) = elems[i]
                self.assertEqual([x], x_val)
                self.assertEqual([y], y_val)

    def testQueueSizeEmpty(self):
        if False:
            print('Hello World!')
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
            self.assertEqual([0], self.evaluate(q.size()))

    def testQueueSizeAfterEnqueueAndDequeue(self):
        if False:
            i = 10
            return i + 15
        with self.session(), self.test_scope():
            q = data_flow_ops.FIFOQueue(10, dtypes_lib.float32)
            enqueue_op = q.enqueue((10.0,))
            dequeued_t = q.dequeue()
            size = q.size()
            self.assertEqual([], size.get_shape())
            enqueue_op.run()
            self.assertEqual(1, self.evaluate(size))
            dequeued_t.op.run()
            self.assertEqual(0, self.evaluate(size))
if __name__ == '__main__':
    test.main()