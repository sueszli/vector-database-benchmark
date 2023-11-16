"""Simple benchmarks for reductions and their gradients."""
import time
import numpy as np
from tensorflow.core.protobuf import config_pb2
from tensorflow.python.client import session
from tensorflow.python.eager import backprop
from tensorflow.python.eager import context
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gradients_impl
from tensorflow.python.ops import math_ops
from tensorflow.python.platform import test

class ReduceBenchmarks(test.Benchmark):
    """Benchmarks for reductions."""

    def _run(self, func, num_iters):
        if False:
            i = 10
            return i + 15
        func()
        start = time.time()
        for _ in range(num_iters):
            func()
        end = time.time()
        mean_us = (end - start) * 1000000.0 / num_iters
        self.report_benchmark(iters=num_iters, wall_time=mean_us, extras={'examples_per_sec': num_iters / (end - start)})

    def benchmark_reduce_sum_grad_eager(self):
        if False:
            print('Hello World!')
        with context.eager_mode():
            tensor = array_ops.zeros([100, 1000])

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)
            self._run(fn, 10000)

    def benchmark_reduce_sum_grad_eager_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        with context.eager_mode(), ops.device('/cpu:0'):
            tensor = array_ops.zeros([100, 1000])

            def fn():
                if False:
                    return 10
                backprop.gradients_function(math_ops.reduce_sum, [0])(tensor)
            self._run(fn, 10000)

    def benchmark_reduce_sum_grad_graph(self):
        if False:
            for i in range(10):
                print('nop')
        config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(optimizer_options=config_pb2.OptimizerOptions(opt_level=config_pb2.OptimizerOptions.L0)))
        with ops.Graph().as_default(), session.Session(config=config) as sess:
            tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
            reduction = math_ops.reduce_sum(tensor)
            (grad,) = gradients_impl.gradients(reduction, tensor)

            def fn():
                if False:
                    for i in range(10):
                        print('nop')
                self.evaluate(grad.op)
            self._run(fn, 10000)

    def benchmark_reduce_sum_grad_graph_cpu(self):
        if False:
            while True:
                i = 10
        config = config_pb2.ConfigProto(graph_options=config_pb2.GraphOptions(optimizer_options=config_pb2.OptimizerOptions(opt_level=config_pb2.OptimizerOptions.L0)))
        with ops.Graph().as_default(), session.Session(config=config) as sess:
            with ops.device('/cpu:0'):
                tensor = constant_op.constant(np.zeros([100, 1000], dtype=np.float32))
                reduction = math_ops.reduce_sum(tensor)
                (grad,) = gradients_impl.gradients(reduction, tensor)

            def fn():
                if False:
                    while True:
                        i = 10
                self.evaluate(grad.op)
            self._run(fn, 10000)
if __name__ == '__main__':
    test.main()