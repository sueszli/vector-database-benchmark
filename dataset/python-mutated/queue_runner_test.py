"""Tests for QueueRunner."""
import collections
import time
from tensorflow.python.client import session
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import errors_impl
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import variable_v1
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import coordinator
from tensorflow.python.training import monitored_session
from tensorflow.python.training import queue_runner_impl
_MockOp = collections.namedtuple('MockOp', ['name'])

@test_util.run_v1_only('QueueRunner removed from v2')
class QueueRunnerTest(test.TestCase):

    def testBasic(self):
        if False:
            return 10
        with self.cached_session() as sess:
            zero64 = constant_op.constant(0, dtype=dtypes.int64)
            var = variable_v1.VariableV1(zero64)
            count_up_to = var.count_up_to(3)
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            self.evaluate(variables.global_variables_initializer())
            qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
            threads = qr.create_threads(sess)
            self.assertEqual(sorted((t.name for t in threads)), ['QueueRunnerThread-fifo_queue-CountUpTo:0'])
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(0, len(qr.exceptions_raised))
            self.assertEqual(3, self.evaluate(var))

    def testTwoOps(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            zero64 = constant_op.constant(0, dtype=dtypes.int64)
            var0 = variable_v1.VariableV1(zero64)
            count_up_to_3 = var0.count_up_to(3)
            var1 = variable_v1.VariableV1(zero64)
            count_up_to_30 = var1.count_up_to(30)
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            qr = queue_runner_impl.QueueRunner(queue, [count_up_to_3, count_up_to_30])
            threads = qr.create_threads(sess)
            self.assertEqual(sorted((t.name for t in threads)), ['QueueRunnerThread-fifo_queue-CountUpTo:0', 'QueueRunnerThread-fifo_queue-CountUpTo_1:0'])
            self.evaluate(variables.global_variables_initializer())
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            self.assertEqual(0, len(qr.exceptions_raised))
            self.assertEqual(3, self.evaluate(var0))
            self.assertEqual(30, self.evaluate(var1))

    def testExceptionsCaptured(self):
        if False:
            for i in range(10):
                print('nop')
        with self.cached_session() as sess:
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            qr = queue_runner_impl.QueueRunner(queue, [_MockOp('i fail'), _MockOp('so fail')])
            threads = qr.create_threads(sess)
            self.evaluate(variables.global_variables_initializer())
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            exceptions = qr.exceptions_raised
            self.assertEqual(2, len(exceptions))
            self.assertTrue('Operation not in the graph' in str(exceptions[0]))
            self.assertTrue('Operation not in the graph' in str(exceptions[1]))

    def testRealDequeueEnqueue(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            q0 = data_flow_ops.FIFOQueue(3, dtypes.float32)
            enqueue0 = q0.enqueue((10.0,))
            close0 = q0.close()
            q1 = data_flow_ops.FIFOQueue(30, dtypes.float32)
            enqueue1 = q1.enqueue((q0.dequeue(),))
            dequeue1 = q1.dequeue()
            qr = queue_runner_impl.QueueRunner(q1, [enqueue1])
            threads = qr.create_threads(sess)
            for t in threads:
                t.start()
            enqueue0.run()
            enqueue0.run()
            close0.run()
            for t in threads:
                t.join()
            self.assertEqual(0, len(qr.exceptions_raised))
            self.assertEqual(10.0, self.evaluate(dequeue1))
            self.assertEqual(10.0, self.evaluate(dequeue1))
            with self.assertRaisesRegex(errors_impl.OutOfRangeError, 'is closed'):
                self.evaluate(dequeue1)

    def testRespectCoordShouldStop(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            zero64 = constant_op.constant(0, dtype=dtypes.int64)
            var = variable_v1.VariableV1(zero64)
            count_up_to = var.count_up_to(3)
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            self.evaluate(variables.global_variables_initializer())
            qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
            coord = coordinator.Coordinator()
            coord.request_stop()
            threads = qr.create_threads(sess, coord)
            self.assertEqual(sorted((t.name for t in threads)), ['QueueRunnerThread-fifo_queue-CountUpTo:0', 'QueueRunnerThread-fifo_queue-close_on_stop'])
            for t in threads:
                t.start()
            coord.join()
            self.assertEqual(0, len(qr.exceptions_raised))
            self.assertEqual(0, self.evaluate(var))

    def testRequestStopOnException(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            qr = queue_runner_impl.QueueRunner(queue, [_MockOp('not an op')])
            coord = coordinator.Coordinator()
            threads = qr.create_threads(sess, coord)
            for t in threads:
                t.start()
            with self.assertRaisesRegex(ValueError, 'Operation not in the graph'):
                coord.join()

    def testGracePeriod(self):
        if False:
            return 10
        with self.cached_session() as sess:
            queue = data_flow_ops.FIFOQueue(2, dtypes.float32)
            enqueue = queue.enqueue((10.0,))
            dequeue = queue.dequeue()
            qr = queue_runner_impl.QueueRunner(queue, [enqueue])
            coord = coordinator.Coordinator()
            qr.create_threads(sess, coord, start=True)
            dequeue.op.run()
            time.sleep(0.02)
            coord.request_stop()
            coord.join(stop_grace_period_secs=1.0)

    def testMultipleSessions(self):
        if False:
            return 10
        with self.cached_session() as sess:
            with session.Session() as other_sess:
                zero64 = constant_op.constant(0, dtype=dtypes.int64)
                var = variable_v1.VariableV1(zero64)
                count_up_to = var.count_up_to(3)
                queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
                self.evaluate(variables.global_variables_initializer())
                coord = coordinator.Coordinator()
                qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
                threads = qr.create_threads(sess, coord=coord)
                other_threads = qr.create_threads(other_sess, coord=coord)
                self.assertEqual(len(threads), len(other_threads))

    def testIgnoreMultiStarts(self):
        if False:
            print('Hello World!')
        with self.cached_session() as sess:
            zero64 = constant_op.constant(0, dtype=dtypes.int64)
            var = variable_v1.VariableV1(zero64)
            count_up_to = var.count_up_to(3)
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            self.evaluate(variables.global_variables_initializer())
            coord = coordinator.Coordinator()
            qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
            threads = []
            threads.extend(qr.create_threads(sess, coord=coord))
            new_threads = qr.create_threads(sess, coord=coord)
            self.assertEqual([], new_threads)

    def testThreads(self):
        if False:
            while True:
                i = 10
        with self.cached_session() as sess:
            zero64 = constant_op.constant(0, dtype=dtypes.int64)
            var = variable_v1.VariableV1(zero64)
            count_up_to = var.count_up_to(3)
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            self.evaluate(variables.global_variables_initializer())
            qr = queue_runner_impl.QueueRunner(queue, [count_up_to, _MockOp('bad_op')])
            threads = qr.create_threads(sess, start=True)
            self.assertEqual(sorted((t.name for t in threads)), ['QueueRunnerThread-fifo_queue-CountUpTo:0', 'QueueRunnerThread-fifo_queue-bad_op'])
            for t in threads:
                t.join()
            exceptions = qr.exceptions_raised
            self.assertEqual(1, len(exceptions))
            self.assertTrue('Operation not in the graph' in str(exceptions[0]))
            threads = qr.create_threads(sess, start=True)
            for t in threads:
                t.join()
            exceptions = qr.exceptions_raised
            self.assertEqual(1, len(exceptions))
            self.assertTrue('Operation not in the graph' in str(exceptions[0]))

    def testName(self):
        if False:
            print('Hello World!')
        with ops.name_scope('scope'):
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32, name='queue')
        qr = queue_runner_impl.QueueRunner(queue, [control_flow_ops.no_op()])
        self.assertEqual('scope/queue', qr.name)
        queue_runner_impl.add_queue_runner(qr)
        self.assertEqual(1, len(ops.get_collection(ops.GraphKeys.QUEUE_RUNNERS, 'scope')))

    def testStartQueueRunners(self):
        if False:
            i = 10
            return i + 15
        zero64 = constant_op.constant(0, dtype=dtypes.int64)
        var = variable_v1.VariableV1(zero64)
        count_up_to = var.count_up_to(3)
        queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
        init_op = variables.global_variables_initializer()
        qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
        queue_runner_impl.add_queue_runner(qr)
        with self.cached_session() as sess:
            init_op.run()
            threads = queue_runner_impl.start_queue_runners(sess)
            for t in threads:
                t.join()
            self.assertEqual(0, len(qr.exceptions_raised))
            self.assertEqual(3, self.evaluate(var))

    def testStartQueueRunnersRaisesIfNotASession(self):
        if False:
            for i in range(10):
                print('nop')
        zero64 = constant_op.constant(0, dtype=dtypes.int64)
        var = variable_v1.VariableV1(zero64)
        count_up_to = var.count_up_to(3)
        queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
        init_op = variables.global_variables_initializer()
        qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
        queue_runner_impl.add_queue_runner(qr)
        with self.cached_session():
            init_op.run()
            with self.assertRaisesRegex(TypeError, 'tf.Session'):
                queue_runner_impl.start_queue_runners('NotASession')

    def testStartQueueRunnersIgnoresMonitoredSession(self):
        if False:
            print('Hello World!')
        zero64 = constant_op.constant(0, dtype=dtypes.int64)
        var = variable_v1.VariableV1(zero64)
        count_up_to = var.count_up_to(3)
        queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
        init_op = variables.global_variables_initializer()
        qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
        queue_runner_impl.add_queue_runner(qr)
        with self.cached_session():
            init_op.run()
            threads = queue_runner_impl.start_queue_runners(monitored_session.MonitoredSession())
            self.assertFalse(threads)

    def testStartQueueRunnersNonDefaultGraph(self):
        if False:
            while True:
                i = 10
        graph = ops.Graph()
        with graph.as_default():
            zero64 = constant_op.constant(0, dtype=dtypes.int64)
            var = variable_v1.VariableV1(zero64)
            count_up_to = var.count_up_to(3)
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32)
            init_op = variables.global_variables_initializer()
            qr = queue_runner_impl.QueueRunner(queue, [count_up_to])
            queue_runner_impl.add_queue_runner(qr)
        with self.session(graph=graph) as sess:
            init_op.run()
            threads = queue_runner_impl.start_queue_runners(sess)
            for t in threads:
                t.join()
            self.assertEqual(0, len(qr.exceptions_raised))
            self.assertEqual(3, self.evaluate(var))

    def testQueueRunnerSerializationRoundTrip(self):
        if False:
            print('Hello World!')
        graph = ops.Graph()
        with graph.as_default():
            queue = data_flow_ops.FIFOQueue(10, dtypes.float32, name='queue')
            enqueue_op = control_flow_ops.no_op(name='enqueue')
            close_op = control_flow_ops.no_op(name='close')
            cancel_op = control_flow_ops.no_op(name='cancel')
            qr0 = queue_runner_impl.QueueRunner(queue, [enqueue_op], close_op, cancel_op, queue_closed_exception_types=(errors_impl.OutOfRangeError, errors_impl.CancelledError))
            qr0_proto = queue_runner_impl.QueueRunner.to_proto(qr0)
            qr0_recon = queue_runner_impl.QueueRunner.from_proto(qr0_proto)
            self.assertEqual('queue', qr0_recon.queue.name)
            self.assertEqual(1, len(qr0_recon.enqueue_ops))
            self.assertEqual(enqueue_op, qr0_recon.enqueue_ops[0])
            self.assertEqual(close_op, qr0_recon.close_op)
            self.assertEqual(cancel_op, qr0_recon.cancel_op)
            self.assertEqual((errors_impl.OutOfRangeError, errors_impl.CancelledError), qr0_recon.queue_closed_exception_types)
            del qr0_proto.queue_closed_exception_types[:]
            qr0_legacy_recon = queue_runner_impl.QueueRunner.from_proto(qr0_proto)
            self.assertEqual('queue', qr0_legacy_recon.queue.name)
            self.assertEqual(1, len(qr0_legacy_recon.enqueue_ops))
            self.assertEqual(enqueue_op, qr0_legacy_recon.enqueue_ops[0])
            self.assertEqual(close_op, qr0_legacy_recon.close_op)
            self.assertEqual(cancel_op, qr0_legacy_recon.cancel_op)
            self.assertEqual((errors_impl.OutOfRangeError,), qr0_legacy_recon.queue_closed_exception_types)
if __name__ == '__main__':
    test.main()