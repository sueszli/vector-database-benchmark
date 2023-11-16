"""Tests for wrapping an eager op in a call op at runtime."""
import time
from tensorflow.python.data.experimental.ops import prefetching_ops
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.eager import benchmarks_test_base
from tensorflow.python.eager import context
from tensorflow.python.eager import test
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import bitwise_ops
from tensorflow.python.ops import critical_section_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.ragged import ragged_factory_ops
from tensorflow.python.ops.ragged import ragged_map_ops
from tensorflow.python.ops.ragged import ragged_tensor
from tensorflow.python.util import tf_inspect

def run_benchmark(func, num_iters, unused_execution_mode):
    if False:
        for i in range(10):
            print('nop')
    func()
    start = time.time()
    for _ in range(num_iters):
        func()
    end = time.time()
    return end - start
CPU = '/device:CPU:0'
GPU = '/device:GPU:0'

@test_util.with_eager_op_as_function
class MicroBenchmarks(benchmarks_test_base.MicroBenchmarksBase):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        super().__init__()
        self._m_2_by_2 = random_ops.random_uniform((2, 2))
        self._m_2_by_2_int32 = random_ops.random_uniform((2, 2), maxval=5, dtype=dtypes.int32)
        self._m_100_by_100 = random_ops.random_uniform((100, 100))
        self._m_100_by_100_int32 = random_ops.random_uniform((100, 100), maxval=5, dtype=dtypes.int32)
        self._m_1000_by_1000 = random_ops.random_uniform((1000, 1000))
        self._m_1000_by_1000_int32 = random_ops.random_uniform((1000, 1000), maxval=5, dtype=dtypes.int32)

    def _get_benchmark_name(self):
        if False:
            return 10
        'Copied from benchmarks_test.py.'
        stack = tf_inspect.stack()
        name = None
        for frame in stack[::-1]:
            f_locals = frame[0].f_locals
            f_self = f_locals.get('self', None)
            if isinstance(f_self, test.Benchmark):
                name = frame[3]
                if name == 'decorated':
                    continue
                else:
                    break
        if name is None:
            raise ValueError('Unable to determine calling Benchmark function.')
        if context.is_tfrt_enabled():
            name = name + '_tfrt'
        if context.run_eager_op_as_function_enabled():
            name = name + '_eager_op_as_function'
        return name

    def _run(self, func, num_iters):
        if False:
            for i in range(10):
                print('nop')
        self.run_report(run_benchmark, func, num_iters, report_mean_us=True)

    def _benchmark_matmul(self, mat, device):
        if False:
            for i in range(10):
                print('nop')
        if device == GPU and (not context.num_gpus()):
            return
        with context.device(device):
            if device == GPU:
                mat = mat.gpu()
            func = lambda : math_ops.matmul(mat, mat)
            self._run(func, num_iters=5000)

    def _benchmark_bitwise_and(self, mat, device):
        if False:
            while True:
                i = 10
        if device == GPU and (not context.num_gpus()):
            return
        with context.device(device):
            if device == GPU:
                mat = mat.gpu()
            func = lambda : bitwise_ops.bitwise_and(mat, mat)
            self._run(func, num_iters=5000)

    def _benchmark_random_normal(self, device):
        if False:
            print('Hello World!')
        if device == GPU and (not context.num_gpus()):
            return
        with context.device(device):

            def func():
                if False:
                    print('Hello World!')
                mat = constant_op.constant([3], dtypes.int32)
                s = mat + mat
                random_ops.random_normal(shape=s)
            self._run(func, num_iters=5000)

    def benchmark_random_normal_GPU(self):
        if False:
            return 10
        self._benchmark_random_normal(GPU)

    def benchmark_tf_matmul_2_by_2_CPU(self):
        if False:
            print('Hello World!')
        self._benchmark_matmul(self._m_2_by_2, CPU)

    def benchmark_tf_bitwise_and_2_by_2_CPU(self):
        if False:
            print('Hello World!')
        self._benchmark_bitwise_and(self._m_2_by_2_int32, CPU)

    def benchmark_tf_matmul_2_by_2_GPU(self):
        if False:
            while True:
                i = 10
        self._benchmark_matmul(self._m_2_by_2, GPU)

    def benchmark_tf_bitwise_and_2_by_2_GPU(self):
        if False:
            print('Hello World!')
        self._benchmark_bitwise_and(self._m_2_by_2_int32, GPU)

    def benchmark_tf_matmul_100_by_100_CPU(self):
        if False:
            return 10
        self._benchmark_matmul(self._m_100_by_100, CPU)

    def benchmark_tf_bitwise_and_100_by_100_CPU(self):
        if False:
            print('Hello World!')
        self._benchmark_bitwise_and(self._m_100_by_100_int32, CPU)

    def benchmark_tf_matmul_100_by_100_GPU(self):
        if False:
            while True:
                i = 10
        self._benchmark_matmul(self._m_100_by_100, GPU)

    def benchmark_tf_bitwise_and_100_by_100_GPU(self):
        if False:
            while True:
                i = 10
        self._benchmark_bitwise_and(self._m_100_by_100_int32, GPU)

    def benchmark_tf_matmul_1000_by_1000_CPU(self):
        if False:
            while True:
                i = 10
        self._benchmark_matmul(self._m_1000_by_1000, CPU)

    def benchmark_tf_bitwise_and_1000_by_1000_CPU(self):
        if False:
            while True:
                i = 10
        self._benchmark_bitwise_and(self._m_1000_by_1000_int32, CPU)

    def benchmark_tf_matmul_1000_by_1000_GPU(self):
        if False:
            print('Hello World!')
        self._benchmark_matmul(self._m_1000_by_1000, GPU)

    def benchmark_tf_bitwise_and_1000_by_1000_GPU(self):
        if False:
            return 10
        self._benchmark_bitwise_and(self._m_1000_by_1000_int32, GPU)

@test_util.with_eager_op_as_function
class RunEagerOpAsFunctionTest(test.TestCase):

    def setUp(self):
        if False:
            for i in range(10):
                print('nop')
        super().setUp()
        self._m_2_by_2 = random_ops.random_uniform((2, 2))

    def testDefaultAttrValues(self):
        if False:
            print('Hello World!')
        ragged_map_ops.map_fn(fn=lambda x: x, elems=ragged_factory_ops.constant([[7]]), dtype=ragged_tensor.RaggedTensorType(dtype=dtypes.int32, ragged_rank=1))

    def testArrayFill(self):
        if False:
            while True:
                i = 10
        array_ops.fill(constant_op.constant([2], dtype=dtypes.int64), constant_op.constant(1))

    def testDatasetMap(self):
        if False:
            i = 10
            return i + 15
        dataset_ops.Dataset.range(2).map(math_ops.square)

    def testPrefetchToDevice(self):
        if False:
            i = 10
            return i + 15
        if not context.num_gpus():
            self.skipTest('No GPU available')
        dataset = dataset_ops.Dataset.range(10)
        dataset = dataset.apply(prefetching_ops.prefetch_to_device('/gpu:0'))

    def testMatmul(self):
        if False:
            i = 10
            return i + 15
        math_ops.matmul(self._m_2_by_2, self._m_2_by_2)

    def testMixedTypeListInputFastPath(self):
        if False:
            return 10
        array_ops.identity_n([self._m_2_by_2, self._m_2_by_2])

    def testMixedTypeListInputEagerFallback(self):
        if False:
            for i in range(10):
                print('nop')
        array_ops.identity_n([1, 1])

    def testMixedTypeListInputFastPathDifferentArity(self):
        if False:
            while True:
                i = 10
        array_ops.identity_n([self._m_2_by_2, self._m_2_by_2])
        array_ops.identity_n([self._m_2_by_2, self._m_2_by_2, self._m_2_by_2])

    def testMixedTypeListInputEagerFallbackDifferentArity(self):
        if False:
            i = 10
            return i + 15
        array_ops.identity_n([1, 1])
        array_ops.identity_n([1, 1, 1])

    def testSingleTypeListFastPath(self):
        if False:
            i = 10
            return i + 15
        array_ops.concat([self._m_2_by_2, self._m_2_by_2], axis=-1)

    def testSingleTypeListEagerFallback(self):
        if False:
            while True:
                i = 10
        array_ops.concat([[1], [2]], axis=-1)

    def testSingleTypeListFastPathDifferentArity(self):
        if False:
            return 10
        array_ops.concat([self._m_2_by_2, self._m_2_by_2], axis=-1)
        array_ops.concat([self._m_2_by_2, self._m_2_by_2, self._m_2_by_2], axis=-1)

    def testSingleTypeListEagerFallbackDifferentArity(self):
        if False:
            print('Hello World!')
        array_ops.concat([[1], [2]], axis=-1)
        array_ops.concat([[1], [2], [3]], axis=-1)

    def testCreateCriticalSection(self):
        if False:
            for i in range(10):
                print('nop')
        cs = critical_section_ops.CriticalSection(shared_name='cs')
        cs.execute(lambda : 1.0)

class RunEagerOpAsFunctionInternalsTest(test.TestCase):

    @test_util.enable_eager_op_as_function
    def testSimpleGraphExecutesSynchronously(self):
        if False:
            while True:
                i = 10
        if context.num_gpus():
            self.skipTest('CPU-only test (requires unpartitioned graph).')
        default_executor = test_util.TestDelta('flr_executor', 'default')
        single_threaded = test_util.TestDelta('flr_executor', 'single_threaded')
        run_async = test_util.TestDelta('pflr_runsync', 'async')
        run_sync = test_util.TestDelta('pflr_runsync', 'sync')
        safe = test_util.TestDelta('subgraph_async_summary', 'safe_for_sync')
        array_ops.fill([2], constant_op.constant(7, dtype=dtypes.int64))
        assert default_executor.Get() == 0
        assert single_threaded.Get() > 0
        assert run_async.Get() == 0
        assert run_sync.Get() > 0
        assert safe.Get() > 0

    @test_util.enable_eager_op_as_function
    def testSendRecvPartitionedGraphExecutesSynchronously(self):
        if False:
            i = 10
            return i + 15
        if not context.num_gpus():
            self.skipTest('GPU-only test (requires partitioned graph).')
        default_executor = test_util.TestDelta('flr_executor', 'default')
        single_threaded = test_util.TestDelta('flr_executor', 'single_threaded')
        run_async = test_util.TestDelta('pflr_runsync', 'async')
        run_sync = test_util.TestDelta('pflr_runsync', 'sync')
        send_only = test_util.TestDelta('subgraph_async_summary', 'send_only')
        recv_only = test_util.TestDelta('subgraph_async_summary', 'recv_only')
        array_ops.fill([2], constant_op.constant(7, dtype=dtypes.int64))
        assert default_executor.Get() == 0
        assert single_threaded.Get() > 0
        assert run_async.Get() == 0
        assert run_sync.Get() > 0
        assert send_only.Get() > 0
        assert recv_only.Get() > 0
if __name__ == '__main__':
    test.main()