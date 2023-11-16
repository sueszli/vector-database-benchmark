"""Benchmark for accumulate_n() in math_ops."""
import random
import time
from tensorflow.python.client import session
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import data_flow_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_state_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import state_ops
from tensorflow.python.platform import test

class AccumulateNBenchmark(test.Benchmark):

    def _AccumulateNTemplate(self, inputs, init, shape, validate_shape):
        if False:
            i = 10
            return i + 15
        var = gen_state_ops.temporary_variable(shape=shape, dtype=inputs[0].dtype.base_dtype)
        ref = state_ops.assign(var, init, validate_shape=validate_shape)
        update_ops = [state_ops.assign_add(ref, tensor, use_locking=True).op for tensor in inputs]
        with ops.control_dependencies(update_ops):
            return gen_state_ops.destroy_temporary_variable(ref, var_name=var.op.name)

    def _AccumulateNInitializedWithFirst(self, inputs):
        if False:
            print('Hello World!')
        return self._AccumulateNTemplate(inputs, init=array_ops.zeros_like(inputs[0]), shape=inputs[0].get_shape(), validate_shape=True)

    def _AccumulateNInitializedWithMerge(self, inputs):
        if False:
            while True:
                i = 10
        return self._AccumulateNTemplate(inputs, init=array_ops.zeros_like(gen_control_flow_ops.merge(inputs)[0]), shape=tensor_shape.TensorShape([0]), validate_shape=False)

    def _AccumulateNInitializedWithShape(self, inputs):
        if False:
            return 10
        return self._AccumulateNTemplate(inputs, init=array_ops.zeros(shape=inputs[0].get_shape(), dtype=inputs[0].dtype.base_dtype), shape=inputs[0].get_shape(), validate_shape=True)

    def _GenerateUnorderedInputs(self, size, n):
        if False:
            for i in range(10):
                print('nop')
        inputs = [random_ops.random_uniform(shape=[size]) for _ in range(n)]
        random.shuffle(inputs)
        return inputs

    def _GenerateReplicatedInputs(self, size, n):
        if False:
            print('Hello World!')
        return n * self._GenerateUnorderedInputs(size, 1)

    def _GenerateOrderedInputs(self, size, n):
        if False:
            while True:
                i = 10
        inputs = self._GenerateUnorderedInputs(size, 1)
        queue = data_flow_ops.FIFOQueue(capacity=1, dtypes=[inputs[0].dtype], shapes=[inputs[0].get_shape()])
        for _ in range(n - 1):
            op = queue.enqueue(inputs[-1])
            with ops.control_dependencies([op]):
                inputs.append(math_ops.tanh(1.0 + queue.dequeue()))
        return inputs

    def _GenerateReversedInputs(self, size, n):
        if False:
            print('Hello World!')
        inputs = self._GenerateOrderedInputs(size, n)
        inputs.reverse()
        return inputs

    def _SetupAndRunBenchmark(self, graph, inputs, repeats, format_args):
        if False:
            for i in range(10):
                print('nop')
        with graph.as_default():
            add_n = math_ops.add_n(inputs)
            acc_n_first = self._AccumulateNInitializedWithFirst(inputs)
            acc_n_merge = self._AccumulateNInitializedWithMerge(inputs)
            acc_n_shape = self._AccumulateNInitializedWithShape(inputs)
        test_ops = (('AddN', add_n.op), ('AccNFirst', acc_n_first.op), ('AccNMerge', acc_n_merge.op), ('AccNShape', acc_n_shape.op))
        with session.Session(graph=graph):
            for (tag, op) in test_ops:
                for _ in range(100):
                    op.run()
                start = time.time()
                for _ in range(repeats):
                    op.run()
                duration = time.time() - start
                args = format_args + (tag, duration)
                print(self._template.format(*args))

    def _RunBenchmark(self, tag, input_fn, sizes, ninputs, repeats):
        if False:
            return 10
        for size in sizes:
            for ninput in ninputs:
                graph = ops.Graph()
                with graph.as_default():
                    inputs = input_fn(size, ninput)
                format_args = (tag, size, ninput, repeats)
                self._SetupAndRunBenchmark(graph, inputs, repeats, format_args)

    def benchmarkAccumulateN(self):
        if False:
            for i in range(10):
                print('nop')
        self._template = '{:<15}' * 6
        args = {'sizes': (128, 128 ** 2), 'ninputs': (1, 10, 100, 300), 'repeats': 100}
        benchmarks = (('Replicated', self._GenerateReplicatedInputs), ('Unordered', self._GenerateUnorderedInputs), ('Ordered', self._GenerateOrderedInputs), ('Reversed', self._GenerateReversedInputs))
        print(self._template.format('', 'Size', '#Inputs', '#Repeat', 'Method', 'Duration'))
        print('-' * 90)
        for benchmark in benchmarks:
            self._RunBenchmark(*benchmark, **args)
if __name__ == '__main__':
    test.main()