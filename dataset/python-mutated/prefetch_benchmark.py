"""Benchmarks for `tf.data.Dataset.prefetch()`."""
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops

class PrefetchBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.Dataset.prefetch()`."""

    def benchmark_prefetch(self):
        if False:
            i = 10
            return i + 15
        num_elements = 1000000
        for prefetch_buffer in [1, 5, 10, 20, 100]:
            dataset = dataset_ops.Dataset.range(num_elements)
            dataset = dataset.prefetch(prefetch_buffer)
            self.run_and_report_benchmark(dataset, num_elements=num_elements, extras={'model_name': 'prefetch.benchmark.1', 'parameters': '%d' % prefetch_buffer}, name='prefetch_{}'.format(prefetch_buffer))
if __name__ == '__main__':
    benchmark_base.test.main()