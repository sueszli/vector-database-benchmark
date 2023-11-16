"""Tests for adding ops to a graph."""
import timeit
from tensorflow.core.config import flags
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import func_graph
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_math_ops
from tensorflow.python.ops import gen_resource_variable_ops
from tensorflow.python.ops import resource_variable_ops
from tensorflow.python.platform import test

@test_util.add_graph_building_optimization_tests
class GraphBuildingBenchmark(test.Benchmark):

    def _computeAddOpDuration(self, num_ops, num_iters):
        if False:
            for i in range(10):
                print('nop')

        def add_op_to_graph(num_ops):
            if False:
                return 10
            with func_graph.FuncGraph('add').as_default():
                a = gen_array_ops.placeholder(dtypes.float32)
                b = gen_array_ops.placeholder(dtypes.float32)
                for _ in range(num_ops):
                    gen_math_ops.add(a, b)
        runtimes = timeit.repeat(lambda : add_op_to_graph(num_ops), repeat=10, number=num_iters)
        return min(runtimes) / num_iters

    def _computeReadVariableOpDuration(self, num_ops, num_iters):
        if False:
            return 10

        def add_op_to_graph(num_ops):
            if False:
                print('Hello World!')
            with func_graph.FuncGraph('resource').as_default():
                handle = resource_variable_ops.var_handle_op(dtype=dtypes.int32, shape=[])
                resource_variable_ops.assign_variable_op(handle, constant_op.constant(1, dtype=dtypes.int32))
                for _ in range(num_ops):
                    gen_resource_variable_ops.read_variable_op(handle, dtype=dtypes.int32)
        runtimes = timeit.repeat(lambda : add_op_to_graph(num_ops), repeat=10, number=num_iters)
        return min(runtimes) / num_iters

    def benchmarkAddOp(self):
        if False:
            print('Hello World!')
        num_ops = 100
        num_iters = 10
        duration = self._computeAddOpDuration(num_ops, num_iters)
        name = 'BenchmarkAddOp'
        if flags.config().graph_building_optimization.value():
            name += 'WithGraphBuildingOptimization'
        self.report_benchmark(name=name, iters=num_iters, wall_time=duration, extras={'num_ops': num_ops})

    def benchmarkResourceVariableOp(self):
        if False:
            for i in range(10):
                print('nop')
        num_ops = 100
        num_iters = 10
        duration = self._computeReadVariableOpDuration(num_ops, num_iters)
        name = 'BenchmarkReadVariableOp'
        if flags.config().graph_building_optimization.value():
            name += 'WithGraphBuildingOptimization'
        self.report_benchmark(name=name, iters=num_iters, wall_time=duration, extras={'num_ops': num_ops})
if __name__ == '__main__':
    ops.enable_eager_execution()
    test.main()