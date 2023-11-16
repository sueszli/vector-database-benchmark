"""Test utilities for tf.data benchmarking functionality."""
import timeit
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.data.experimental.ops import testing
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.eager import context
from tensorflow.python.platform import test

class MetaBenchmark(test.Benchmark):
    """Benchmark that compares various ways of running tf.data benchmarks."""

    def setup_fast_dataset(self):
        if False:
            return 10
        self.num_reps = 15
        self.iters = 100000
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = False
        return dataset_ops.Dataset.range(10000 ** 2).with_options(options)

    def benchmark_fast_dataset_with_only_cpp_iterations(self):
        if False:
            i = 10
            return i + 15
        dataset = self.setup_fast_dataset()
        self.run_benchmark_with_only_cpp_iterations(dataset)

    def benchmark_fast_dataset_with_session_run(self):
        if False:
            print('Hello World!')
        dataset = self.setup_fast_dataset()
        self.run_benchmark_with_session_run(dataset)

    def benchmark_fast_dataset_with_session_callable(self):
        if False:
            while True:
                i = 10
        dataset = self.setup_fast_dataset()
        self.run_benchmark_with_session_run(dataset, make_callable=True)

    def benchmark_fast_dataset_in_eager(self):
        if False:
            print('Hello World!')
        with context.eager_mode():
            dataset = self.setup_fast_dataset()
            self.run_benchmark_in_eager(dataset)

    def setup_slow_dataset(self):
        if False:
            for i in range(10):
                print('nop')
        dataset = self.setup_fast_dataset()
        self.iters = 1000
        return dataset.apply(testing.sleep(1000))

    def benchmark_slow_dataset_with_only_cpp_iterations(self):
        if False:
            while True:
                i = 10
        dataset = self.setup_slow_dataset()
        self.run_benchmark_with_only_cpp_iterations(dataset)

    def benchmark_slow_dataset_with_session_run(self):
        if False:
            return 10
        dataset = self.setup_slow_dataset()
        self.run_benchmark_with_session_run(dataset)

    def benchmark_slow_dataset_with_session_callable(self):
        if False:
            print('Hello World!')
        dataset = self.setup_slow_dataset()
        self.run_benchmark_with_session_run(dataset, make_callable=True)

    def benchmark_slow_dataset_in_eager(self):
        if False:
            i = 10
            return i + 15
        with context.eager_mode():
            dataset = self.setup_slow_dataset()
            self.run_benchmark_in_eager(dataset)

    def report(self, deltas):
        if False:
            i = 10
            return i + 15
        deltas = np.array(deltas) / self.iters
        deltas = deltas[5:]
        median = np.median(deltas)
        mean = np.mean(deltas)
        min_val = np.min(deltas)
        max_val = np.max(deltas)
        extras = {'iters_per_second': 1 / median, 'median': median, 'mean': mean, 'min': min_val, 'max': max_val, 'num_reps': self.num_reps - 5}
        self.report_benchmark(wall_time=median, iters=self.iters, extras=extras)

    def run_benchmark_in_eager(self, dataset):
        if False:
            i = 10
            return i + 15
        deltas = []
        for _ in range(self.num_reps):
            iterator = iter(dataset)
            deltas.append(timeit.timeit(lambda : next(iterator), number=self.iters))
        self.report(deltas)

    def run_benchmark_with_session_run(self, dataset, make_callable=False):
        if False:
            return 10
        iterator = dataset_ops.make_initializable_iterator(dataset)
        next_element = iterator.get_next()
        with session.Session() as sess:
            deltas = []
            for _ in range(self.num_reps):
                if make_callable:
                    get_next_element = sess.make_callable(next_element)
                else:
                    get_next_element = lambda : sess.run(next_element.op)
                sess.run(iterator.initializer)
                deltas.append(timeit.timeit(get_next_element, number=self.iters))
        self.report(deltas)

    def run_benchmark_with_only_cpp_iterations(self, dataset):
        if False:
            return 10
        'Benchmarks the dataset with the iterations performed in C++.'
        dataset = dataset.skip(self.iters - 1)
        iterator = dataset_ops.make_initializable_iterator(dataset)
        next_element = iterator.get_next()
        with session.Session() as sess:
            deltas = []
            for _ in range(self.num_reps):
                sess.run(iterator.initializer)
                deltas.append(timeit.timeit(lambda : sess.run(next_element.op), number=1))
        self.report(deltas)
if __name__ == '__main__':
    test.main()