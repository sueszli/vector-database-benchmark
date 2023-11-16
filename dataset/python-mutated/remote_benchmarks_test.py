"""Benchmarks for remote worker eager execution.

To run CPU benchmarks:
  bazel run -c opt remote_benchmarks_test -- --benchmark_filter=.

To run GPU benchmarks:
  bazel run --config=cuda -c opt --copt="-mavx" remote_benchmarks_test -- \\
    --benchmark_filter=.
"""
import gc
import time
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.eager import remote
from tensorflow.python.eager import test
from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import variables
from tensorflow.python.training import server_lib

def run_benchmark(func, num_iters, execution_mode=None):
    if False:
        for i in range(10):
            print('nop')
    ctx = context.context()
    with context.execution_mode(execution_mode):
        func()
        if execution_mode == context.ASYNC:
            ctx.executor.wait()
        start = time.time()
        for _ in range(num_iters):
            func()
        if execution_mode == context.ASYNC:
            ctx.executor.wait()
        end = time.time()
        return end - start

class Foo(object):

    def __init__(self, num_vars):
        if False:
            while True:
                i = 10
        self._num_vars = num_vars
        self._v = []

    def __call__(self, inputs):
        if False:
            print('Hello World!')
        if not self._v:
            for _ in range(self._num_vars):
                self._v.append(variables.Variable(random_ops.random_uniform([]), shape=[]))
        for v in self._v:
            inputs = inputs * v
        return inputs

class RemoteWorkerMicroBenchmarks(test.Benchmark):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self._cached_server1 = server_lib.Server.create_local_server()
        self._cached_server_target1 = self._cached_server1.target[len('grpc://'):]
        self._cached_server2 = server_lib.Server.create_local_server()
        self._cached_server_target2 = self._cached_server2.target[len('grpc://'):]

    def _run(self, func, num_iters=1000, execution_mode=context.ASYNC):
        if False:
            while True:
                i = 10
        total_time = run_benchmark(func, num_iters, execution_mode)
        mean_us = total_time * 1000000.0 / num_iters
        self.report_benchmark(iters=num_iters, wall_time=mean_us, extras={'examples_per_sec': num_iters / total_time})

    def benchmark_send(self):
        if False:
            print('Hello World!')
        remote.connect_to_remote_host(self._cached_server_target1)
        x = random_ops.random_uniform((2, 2)).cpu()

        @def_function.function
        def remote_func(m):
            if False:
                i = 10
                return i + 15
            return math_ops.matmul(m, m)

        def func(m):
            if False:
                print('Hello World!')
            with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
                return remote_func(m)
        self._run(lambda : func(x))
        gc.collect()

    def benchmark_worker_recv(self):
        if False:
            for i in range(10):
                print('nop')
        remote.connect_to_remote_host([self._cached_server_target1, self._cached_server_target2])
        with ops.device('job:worker/replica:0/task:1/device:CPU:0'):
            v = variables.Variable(1.0)

        @def_function.function
        def remote_func():
            if False:
                return 10
            return 1.0 + v

        def func():
            if False:
                return 10
            with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
                return remote_func()
        self._run(func)
        gc.collect()

    def benchmark_create_vars_inside_function(self):
        if False:
            while True:
                i = 10
        remote.connect_to_remote_host(self._cached_server_target1)

        def func():
            if False:
                i = 10
                return i + 15
            with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
                layer = Foo(50)

                @def_function.function
                def remote_func():
                    if False:
                        print('Hello World!')
                    with ops.device('job:worker/replica:0/task:0/device:CPU:0'):
                        return layer(random_ops.random_uniform([]))
                return remote_func()
        self._run(func, execution_mode=context.ASYNC, num_iters=100)
        gc.collect()
if __name__ == '__main__':
    test.main()