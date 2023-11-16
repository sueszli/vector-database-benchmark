"""Benchmarks for `tf.data.experimental.rejection_resample()`."""
import numpy as np
from tensorflow.python.data.benchmarks import benchmark_base
from tensorflow.python.data.experimental.ops import resampling
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib

class RejectionResampleBenchmark(benchmark_base.DatasetBenchmarkBase):
    """Benchmarks for `tf.data.experimental.rejection_resample()`."""

    def benchmark_resample_performance(self):
        if False:
            i = 10
            return i + 15
        init_dist = [0.25, 0.25, 0.25, 0.25]
        target_dist = [0.0, 0.0, 0.0, 1.0]
        num_classes = len(init_dist)
        num_samples = 1000
        data_np = np.random.choice(num_classes, num_samples, p=init_dist)
        dataset = dataset_ops.Dataset.from_tensor_slices(data_np).repeat()
        dataset = dataset.apply(resampling.rejection_resample(class_func=lambda x: x, target_dist=target_dist, initial_dist=init_dist, seed=142))
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        dataset = dataset.with_options(options)
        wall_time = self.run_benchmark(dataset=dataset, num_elements=num_samples, iters=10, warmup=True)
        resample_time = wall_time * num_samples
        self.report_benchmark(iters=10, wall_time=resample_time, extras={'model_name': 'rejection_resample.benchmark.1', 'parameters': '%d' % num_samples}, name='resample_{}'.format(num_samples))
if __name__ == '__main__':
    benchmark_base.test.main()