"""Benchmarks for `tf.data.Dataset.range()`."""
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib

class RangeBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.Dataset.range()`."""

    def _benchmark_range(self, num_elements, autotune, benchmark_id):
        if False:
            for i in range(10):
                print('nop')
        options = options_lib.Options()
        options.autotune.enabled = autotune
        dataset = dataset_ops.Dataset.range(num_elements)
        dataset = dataset.with_options(options)
        self.run_and_report_benchmark(dataset, num_elements=num_elements, extras={'model_name': 'range.benchmark.%d' % benchmark_id, 'parameters': '%d.%s' % (num_elements, autotune)}, name='modeling_%s' % ('on' if autotune else 'off'))

    def benchmark_range_with_modeling(self):
        if False:
            return 10
        self._benchmark_range(num_elements=10000000, autotune=True, benchmark_id=1)

    def benchmark_range_without_modeling(self):
        if False:
            i = 10
            return i + 15
        self._benchmark_range(num_elements=50000000, autotune=False, benchmark_id=2)
if __name__ == '__main__':
    benchmark_base.test.main()