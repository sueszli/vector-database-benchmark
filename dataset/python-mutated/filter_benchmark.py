"""Benchmarks for `tf.data.Dataset.filter()`."""
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.ops import array_ops

class FilterBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.Dataset.filter()`."""

    def _benchmark(self, predicate, name, benchmark_id):
        if False:
            return 10
        num_elements = 100000
        dataset = dataset_ops.Dataset.from_tensors(True)
        dataset = dataset.repeat().filter(predicate)
        self.run_and_report_benchmark(dataset, num_elements=num_elements, extras={'model_name': 'filter.benchmark.%d' % benchmark_id, 'parameters': '%d' % num_elements}, name=name)

    def benchmark_simple_function(self):
        if False:
            print('Hello World!')
        self._benchmark(array_ops.identity, 'simple_function', benchmark_id=1)

    def benchmark_return_component_optimization(self):
        if False:
            print('Hello World!')
        self._benchmark(lambda x: x, 'return_component', benchmark_id=2)
if __name__ == '__main__':
    benchmark_base.test.main()