"""Tests and benchmarks for interacting with the `tf.compat.v1.Session`."""
import time
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import test
from tensorflow.python.training import server_lib

class SessionBenchmark(test.Benchmark):
    """Tests and benchmarks for interacting with the `tf.compat.v1.Session`."""

    def _benchmarkFeed(self, name, target, size, iters):
        if False:
            while True:
                i = 10
        'Runs a microbenchmark to measure the cost of feeding a tensor.\n\n    Reports the median cost of feeding a tensor of `size` * `sizeof(float)`\n    bytes.\n\n    Args:\n      name: A human-readable name for logging the output.\n      target: The session target to use for the benchmark.\n      size: The number of floating-point numbers to be feed.\n      iters: The number of iterations to perform.\n    '
        feed_val = np.random.rand(size).astype(np.float32)
        times = []
        with ops.Graph().as_default():
            p = array_ops.placeholder(dtypes.float32, shape=[size])
            no_op = array_ops.identity(p).op
            with session.Session(target) as sess:
                sess.run(no_op, feed_dict={p: feed_val})
                for _ in range(iters):
                    start_time = time.time()
                    sess.run(no_op, feed_dict={p: feed_val})
                    end_time = time.time()
                    times.append(end_time - start_time)
        print('%s %d %f' % (name, size, np.median(times)))
        self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

    def _benchmarkFetch(self, name, target, size, iters):
        if False:
            return 10
        'Runs a microbenchmark to measure the cost of fetching a tensor.\n\n    Reports the median cost of fetching a tensor of `size` * `sizeof(float)`\n    bytes.\n\n    Args:\n      name: A human-readable name for logging the output.\n      target: The session target to use for the benchmark.\n      size: The number of floating-point numbers to be fetched.\n      iters: The number of iterations to perform.\n    '
        times = []
        with ops.Graph().as_default():
            v = variables.Variable(random_ops.random_normal([size]))
            with session.Session(target) as sess:
                sess.run(v.initializer)
                sess.run(v)
                for _ in range(iters):
                    start_time = time.time()
                    sess.run(v)
                    end_time = time.time()
                    times.append(end_time - start_time)
        print('%s %d %f' % (name, size, np.median(times)))
        self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

    def _benchmarkFetchPrebuilt(self, name, target, size, iters):
        if False:
            print('Hello World!')
        'Runs a microbenchmark to measure the cost of fetching a tensor.\n\n    Reports the median cost of fetching a tensor of `size` * `sizeof(float)`\n    bytes.\n\n    Args:\n      name: A human-readable name for logging the output.\n      target: The session target to use for the benchmark.\n      size: The number of floating-point numbers to be fetched.\n      iters: The number of iterations to perform.\n    '
        times = []
        with ops.Graph().as_default():
            v = variables.Variable(random_ops.random_normal([size]))
            with session.Session(target) as sess:
                sess.run(v.initializer)
                runner = sess.make_callable(v)
                runner()
                for _ in range(iters):
                    start_time = time.time()
                    runner()
                    end_time = time.time()
                    times.append(end_time - start_time)
        print('%s %d %f' % (name, size, np.median(times)))
        self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

    def _benchmarkRunOp(self, name, target, iters):
        if False:
            while True:
                i = 10
        'Runs a microbenchmark to measure the cost of running an op.\n\n    Reports the median cost of running a trivial (Variable) op.\n\n    Args:\n      name: A human-readable name for logging the output.\n      target: The session target to use for the benchmark.\n      iters: The number of iterations to perform.\n    '
        times = []
        with ops.Graph().as_default():
            v = variables.Variable(random_ops.random_normal([]))
            with session.Session(target) as sess:
                sess.run(v.initializer)
                sess.run(v.op)
                for _ in range(iters):
                    start_time = time.time()
                    sess.run(v.op)
                    end_time = time.time()
                    times.append(end_time - start_time)
        print('%s %f' % (name, np.median(times)))
        self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

    def _benchmarkRunOpPrebuilt(self, name, target, iters):
        if False:
            i = 10
            return i + 15
        'Runs a microbenchmark to measure the cost of running an op.\n\n    Reports the median cost of running a trivial (Variable) op.\n\n    Args:\n      name: A human-readable name for logging the output.\n      target: The session target to use for the benchmark.\n      iters: The number of iterations to perform.\n    '
        times = []
        with ops.Graph().as_default():
            v = variables.Variable(random_ops.random_normal([]))
            with session.Session(target) as sess:
                sess.run(v.initializer)
                runner = sess.make_callable(v.op)
                runner()
                for _ in range(iters):
                    start_time = time.time()
                    runner()
                    end_time = time.time()
                    times.append(end_time - start_time)
        print('%s %f' % (name, np.median(times)))
        self.report_benchmark(iters=1, wall_time=np.median(times), name=name)

    def benchmarkGrpcSession(self):
        if False:
            print('Hello World!')
        server = server_lib.Server.create_local_server()
        self._benchmarkFeed('benchmark_session_feed_grpc_4B', server.target, 1, 30000)
        session.Session.reset(server.target)
        self._benchmarkFeed('benchmark_session_feed_grpc_4MB', server.target, 1 << 20, 25000)
        session.Session.reset(server.target)
        self._benchmarkFetch('benchmark_session_fetch_grpc_4B', server.target, 1, 40000)
        session.Session.reset(server.target)
        self._benchmarkFetch('benchmark_session_fetch_grpc_4MB', server.target, 1 << 20, 20000)
        session.Session.reset(server.target)
        self._benchmarkFetchPrebuilt('benchmark_session_fetchprebuilt_grpc_4B', server.target, 1, 50000)
        session.Session.reset(server.target)
        self._benchmarkFetchPrebuilt('benchmark_session_fetchprebuilt_grpc_4MB', server.target, 1 << 20, 50000)
        session.Session.reset(server.target)
        self._benchmarkRunOp('benchmark_session_runop_grpc', server.target, 50000)
        session.Session.reset(server.target)
        self._benchmarkRunOpPrebuilt('benchmark_session_runopprebuilt_grpc', server.target, 100000)
        session.Session.reset(server.target)

    def benchmarkDirectSession(self):
        if False:
            for i in range(10):
                print('nop')
        self._benchmarkFeed('benchmark_session_feed_direct_4B', '', 1, 80000)
        self._benchmarkFeed('benchmark_session_feed_direct_4MB', '', 1 << 20, 20000)
        self._benchmarkFetch('benchmark_session_fetch_direct_4B', '', 1, 100000)
        self._benchmarkFetch('benchmark_session_fetch_direct_4MB', '', 1 << 20, 20000)
        self._benchmarkFetchPrebuilt('benchmark_session_fetchprebuilt_direct_4B', '', 1, 200000)
        self._benchmarkFetchPrebuilt('benchmark_session_fetchprebuilt_direct_4MB', '', 1 << 20, 200000)
        self._benchmarkRunOp('benchmark_session_runop_direct', '', 200000)
        self._benchmarkRunOpPrebuilt('benchmark_session_runopprebuilt_direct', '', 200000)
if __name__ == '__main__':
    test.main()