"""Benchmarks for static optimizations."""
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.ops import math_ops

class OptimizationBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for static optimizations."""

    def benchmark_map_fusion(self):
        if False:
            for i in range(10):
                print('nop')
        'Evaluates performance map of fusion.'
        chain_lengths = [0, 1, 2, 5, 10, 20, 50]
        for chain_length in chain_lengths:
            self._benchmark_map_fusion(chain_length=chain_length, optimize_dataset=False)
            self._benchmark_map_fusion(chain_length=chain_length, optimize_dataset=True)

    def _benchmark_map_fusion(self, chain_length, optimize_dataset):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
        for _ in range(chain_length):
            dataset = dataset.map(lambda x: x)
        if optimize_dataset:
            options = options_lib.Options()
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_optimization.map_fusion = True
            dataset = dataset.with_options(options)
        opt_mark = 'opt' if optimize_dataset else 'noopt'
        self.run_and_report_benchmark(dataset=dataset, num_elements=100, iters=10, warmup=True, extras={'model_name': 'optimize.benchmark.1', 'parameters': '%d.%s' % (chain_length, optimize_dataset)}, name='map_fusion_{}_chain_length_{}'.format(opt_mark, chain_length))

    def benchmark_map_and_filter_fusion(self):
        if False:
            print('Hello World!')
        'Evaluates performance map of fusion.'
        chain_lengths = [0, 1, 2, 5, 10, 20, 50]
        for chain_length in chain_lengths:
            self._benchmark_map_and_filter_fusion(chain_length=chain_length, optimize_dataset=False)
            self._benchmark_map_and_filter_fusion(chain_length=chain_length, optimize_dataset=True)

    def _benchmark_map_and_filter_fusion(self, chain_length, optimize_dataset):
        if False:
            print('Hello World!')
        dataset = dataset_ops.Dataset.from_tensors(0).repeat(None)
        for _ in range(chain_length):
            dataset = dataset.map(lambda x: x + 5).filter(lambda x: math_ops.greater_equal(x - 5, 0))
        if optimize_dataset:
            options = options_lib.Options()
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_optimization.map_and_filter_fusion = True
            dataset = dataset.with_options(options)
        opt_mark = 'opt' if optimize_dataset else 'noopt'
        self.run_and_report_benchmark(dataset=dataset, num_elements=100, iters=10, warmup=True, extras={'model_name': 'optimize.benchmark.2', 'parameters': '%d.%s' % (chain_length, optimize_dataset)}, name='map_and_filter_fusion_{}_chain_length_{}'.format(opt_mark, chain_length))

    def benchmark_filter_fusion(self):
        if False:
            i = 10
            return i + 15
        chain_lengths = [0, 1, 2, 5, 10, 20, 50]
        for chain_length in chain_lengths:
            self._benchmark_filter_fusion(chain_length=chain_length, optimize_dataset=False)
            self._benchmark_filter_fusion(chain_length=chain_length, optimize_dataset=True)

    def _benchmark_filter_fusion(self, chain_length, optimize_dataset):
        if False:
            i = 10
            return i + 15
        dataset = dataset_ops.Dataset.from_tensors(5).repeat(None)
        for _ in range(chain_length):
            dataset = dataset.filter(lambda x: math_ops.greater_equal(x - 5, 0))
        if optimize_dataset:
            options = options_lib.Options()
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_optimization.filter_fusion = True
            dataset = dataset.with_options(options)
        opt_mark = 'opt' if optimize_dataset else 'noopt'
        self.run_and_report_benchmark(dataset=dataset, num_elements=100, iters=10, warmup=True, extras={'model_name': 'optimize.benchmark.3', 'parameters': '%d.%s' % (chain_length, optimize_dataset)}, name='filter_fusion_{}_chain_length_{}'.format(opt_mark, chain_length))

    def benchmark_filter_parallelization(self):
        if False:
            return 10
        chain_lengths = [0, 1, 2, 5, 10, 20, 50]
        for chain_length in chain_lengths:
            self._benchmark_filter_parallelization(chain_length=chain_length, optimize_dataset=False)
            self._benchmark_filter_parallelization(chain_length=chain_length, optimize_dataset=True)

    def _benchmark_filter_parallelization(self, chain_length, optimize_dataset):
        if False:
            for i in range(10):
                print('nop')
        dataset = dataset_ops.Dataset.from_tensors(5).repeat()
        for _ in range(chain_length):
            dataset = dataset.filter(lambda x: math_ops.greater_equal(x - 5, 0))
        if optimize_dataset:
            options = options_lib.Options()
            options.experimental_optimization.apply_default_optimizations = False
            options.experimental_optimization.filter_parallelization = True
            dataset = dataset.with_options(options)
        opt_mark = 'opt' if optimize_dataset else 'noopt'
        self.run_and_report_benchmark(dataset=dataset, num_elements=100, iters=10, warmup=True, extras={'model_name': 'optimize.benchmark.4', 'parameters': '%d.%s' % (chain_length, optimize_dataset)}, name='filter_parallelization_{}_chain_length_{}'.format(opt_mark, chain_length))
if __name__ == '__main__':
    benchmark_base.test.main()