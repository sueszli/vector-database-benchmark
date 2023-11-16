"""Benchmarks for autotuning performance knobs."""
import numpy as np
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.ops import math_ops

class AutotuneBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for autotuning performance knobs."""

    def _run_benchmark(self, dataset, autotune, benchmark_iters, benchmark_label, benchmark_id):
        if False:
            for i in range(10):
                print('nop')
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        options.autotune.enabled = autotune
        dataset = dataset.with_options(options)
        autotune_string = '_autotune_parallelism_only'
        wall_time = self.run_and_report_benchmark(dataset=dataset, num_elements=benchmark_iters, warmup=True, iters=1, extras={'model_name': 'autotune.benchmark.%s.%d' % (benchmark_label, benchmark_id), 'parameters': '%s' % autotune}, name=benchmark_label + (autotune_string if autotune else ''))
        return wall_time

    def benchmark_batch(self):
        if False:
            while True:
                i = 10
        a = self._benchmark_batch(autotune=False, benchmark_id=1)
        b = self._benchmark_batch(autotune=True, benchmark_id=2)
        print('autotune parallelism vs no autotuning speedup: {}'.format(a / b))

    def _benchmark_batch(self, autotune, benchmark_id):
        if False:
            while True:
                i = 10
        batch_size = 128
        k = 1024
        dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
        dataset = dataset.map(math_ops.matmul)
        dataset = dataset.batch(batch_size=batch_size, num_parallel_calls=dataset_ops.AUTOTUNE)
        return self._run_benchmark(dataset=dataset, autotune=autotune, benchmark_iters=10000, benchmark_label='batch', benchmark_id=benchmark_id)

    def benchmark_map(self):
        if False:
            i = 10
            return i + 15
        a = self._benchmark_map(autotune=False, benchmark_id=1)
        b = self._benchmark_map(autotune=True, benchmark_id=2)
        print('autotune parallelism vs no autotuning speedup: {}'.format(a / b))

    def _benchmark_map(self, autotune, benchmark_id):
        if False:
            while True:
                i = 10
        k = 1024 * 1024
        dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
        dataset = dataset.map(math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
        return self._run_benchmark(dataset=dataset, autotune=autotune, benchmark_iters=10000, benchmark_label='map', benchmark_id=benchmark_id)

    def benchmark_map_and_batch(self):
        if False:
            while True:
                i = 10
        a = self._benchmark_map_and_batch(autotune=False, benchmark_id=1)
        b = self._benchmark_map_and_batch(autotune=True, benchmark_id=2)
        print('autotune parallelism vs no autotuning speedup: {}'.format(a / b))

    def _benchmark_map_and_batch(self, autotune, benchmark_id):
        if False:
            for i in range(10):
                print('nop')
        batch_size = 16
        k = 1024 * 1024
        dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
        dataset = dataset.map(math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size)
        return self._run_benchmark(dataset=dataset, autotune=autotune, benchmark_iters=1000, benchmark_label='map_and_batch', benchmark_id=benchmark_id)

    def benchmark_interleave(self):
        if False:
            while True:
                i = 10
        a = self._benchmark_interleave(autotune=False, benchmark_id=1)
        b = self._benchmark_interleave(autotune=True, benchmark_id=2)
        print('autotune parallelism vs no autotuning speedup: {}'.format(a / b))

    def _benchmark_interleave(self, autotune, benchmark_id):
        if False:
            i = 10
            return i + 15
        k = 1024 * 1024
        dataset = dataset_ops.Dataset.from_tensors((np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))).repeat()
        dataset = dataset.map(math_ops.matmul)
        dataset = dataset_ops.Dataset.range(1).repeat().interleave(lambda _: dataset, cycle_length=10, num_parallel_calls=dataset_ops.AUTOTUNE)
        return self._run_benchmark(dataset=dataset, autotune=autotune, benchmark_iters=10000, benchmark_label='interleave', benchmark_id=benchmark_id)

    def benchmark_map_and_interleave(self):
        if False:
            print('Hello World!')
        a = self._benchmark_map_and_interleave(autotune=False, benchmark_id=1)
        b = self._benchmark_map_and_interleave(autotune=True, benchmark_id=2)
        print('autotune parallelism vs no autotuning speedup: {}'.format(a / b))

    def _benchmark_map_and_interleave(self, autotune, benchmark_id):
        if False:
            i = 10
            return i + 15
        k = 1024 * 1024
        a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
        b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
        c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
        dataset_a = dataset_ops.Dataset.from_tensors(a).repeat()
        dataset_b = dataset_ops.Dataset.from_tensors(b).repeat()
        dataset_c = dataset_ops.Dataset.from_tensors(c).repeat()

        def f1(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.matmul(x, y)

        def f2(a, b):
            if False:
                print('Hello World!')
            (x, y) = b
            return (a, math_ops.matmul(x, y))
        dataset = dataset_a
        dataset = dataset.map(f1, num_parallel_calls=dataset_ops.AUTOTUNE)
        dataset = dataset_ops.Dataset.range(1).repeat().interleave(lambda _: dataset, num_parallel_calls=dataset_ops.AUTOTUNE, cycle_length=2)
        dataset = dataset_ops.Dataset.zip((dataset, dataset_b))
        dataset = dataset.map(f2, num_parallel_calls=dataset_ops.AUTOTUNE)
        dataset = dataset_ops.Dataset.range(1).repeat().interleave(lambda _: dataset, num_parallel_calls=dataset_ops.AUTOTUNE, cycle_length=2)
        dataset = dataset_ops.Dataset.zip((dataset, dataset_c))
        dataset = dataset.map(f2, num_parallel_calls=dataset_ops.AUTOTUNE)
        return self._run_benchmark(dataset=dataset, autotune=autotune, benchmark_iters=10000, benchmark_label='map_and_interleave', benchmark_id=benchmark_id)

    def benchmark_map_batch_and_interleave(self):
        if False:
            return 10
        a = self._benchmark_map_batch_and_interleave(autotune=False, benchmark_id=1)
        b = self._benchmark_map_batch_and_interleave(autotune=True, benchmark_id=2)
        print('autotune parallelism vs no autotuning speedup: {}'.format(a / b))

    def _benchmark_map_batch_and_interleave(self, autotune, benchmark_id):
        if False:
            while True:
                i = 10
        batch_size = 16
        k = 1024 * 1024
        a = (np.random.rand(1, 8 * k), np.random.rand(8 * k, 1))
        b = (np.random.rand(1, 4 * k), np.random.rand(4 * k, 1))
        c = (np.random.rand(1, 2 * k), np.random.rand(2 * k, 1))
        dataset_a = dataset_ops.Dataset.from_tensors(a).repeat()
        dataset_b = dataset_ops.Dataset.from_tensors(b).repeat()
        dataset_c = dataset_ops.Dataset.from_tensors(c).repeat()
        dataset = dataset_a
        dataset = dataset.map(math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
        dataset = dataset.batch(batch_size=batch_size)
        dataset = dataset_ops.Dataset.range(1).repeat().interleave(lambda _: dataset, num_parallel_calls=dataset_ops.AUTOTUNE, cycle_length=2)
        dataset = dataset_ops.Dataset.zip((dataset, dataset_b))
        dataset = dataset_ops.Dataset.range(1).repeat().interleave(lambda _: dataset, num_parallel_calls=dataset_ops.AUTOTUNE, cycle_length=2)
        dataset_c = dataset_c.map(math_ops.matmul, num_parallel_calls=dataset_ops.AUTOTUNE)
        dataset_c = dataset_c.batch(batch_size=batch_size)
        dataset = dataset_ops.Dataset.zip((dataset, dataset_c))
        return self._run_benchmark(dataset=dataset, autotune=autotune, benchmark_iters=1000, benchmark_label='map_and_batch_and_interleave', benchmark_id=benchmark_id)
if __name__ == '__main__':
    benchmark_base.test.main()