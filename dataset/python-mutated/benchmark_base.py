"""Test utilities for tf.data benchmarking functionality."""
import time
import numpy as np
from tensorflow.python.client import session
from tensorflow.python.data.ops import dataset_ops
from tensorflow.python.data.ops import options as options_lib
from tensorflow.python.data.util import nest
from tensorflow.python.eager import context
from tensorflow.python.platform import test

class DatasetBenchmarkBase(test.Benchmark):
    """Base class for dataset benchmarks."""

    def _run_eager_benchmark(self, iterable, iters, warmup):
        if False:
            return 10
        'Benchmark the iterable in eager mode.\n\n    Runs the iterable `iters` times. In each iteration, the benchmark measures\n    the time it takes to go execute the iterable.\n\n    Args:\n      iterable: The tf op or tf.data Dataset to benchmark.\n      iters: Number of times to repeat the timing.\n      warmup: If true, warms up the session caches by running an untimed run.\n\n    Returns:\n      A float, representing the median time (with respect to `iters`)\n      it takes for the iterable to be executed `iters` num of times.\n\n    Raises:\n      RuntimeError: When executed in graph mode.\n    '
        deltas = []
        if not context.executing_eagerly():
            raise RuntimeError('Eager mode benchmarking is not supported in graph mode.')
        for _ in range(iters):
            if warmup:
                iterator = iter(iterable)
                next(iterator)
            iterator = iter(iterable)
            start = time.time()
            next(iterator)
            end = time.time()
            deltas.append(end - start)
        return np.median(deltas)

    def _run_graph_benchmark(self, iterable, iters, warmup, session_config, initializer=None):
        if False:
            i = 10
            return i + 15
        'Benchmarks the iterable in graph mode.\n\n    Runs the iterable `iters` times. In each iteration, the benchmark measures\n    the time it takes to go execute the iterable.\n\n    Args:\n      iterable: The tf op or tf.data Dataset to benchmark.\n      iters: Number of times to repeat the timing.\n      warmup: If true, warms up the session caches by running an untimed run.\n      session_config: A ConfigProto protocol buffer with configuration options\n        for the session. Applicable only for benchmarking in graph mode.\n      initializer: The initializer op required to initialize the iterable.\n\n    Returns:\n      A float, representing the median time (with respect to `iters`)\n      it takes for the iterable to be executed `iters` num of times.\n\n    Raises:\n      RuntimeError: When executed in eager mode.\n    '
        deltas = []
        if context.executing_eagerly():
            raise RuntimeError('Graph mode benchmarking is not supported in eager mode.')
        for _ in range(iters):
            with session.Session(config=session_config) as sess:
                if warmup:
                    if initializer:
                        sess.run(initializer)
                    sess.run(iterable)
                if initializer:
                    sess.run(initializer)
                start = time.time()
                sess.run(iterable)
                end = time.time()
            deltas.append(end - start)
        return np.median(deltas)

    def run_op_benchmark(self, op, iters=1, warmup=True, session_config=None):
        if False:
            for i in range(10):
                print('nop')
        'Benchmarks the op.\n\n    Runs the op `iters` times. In each iteration, the benchmark measures\n    the time it takes to go execute the op.\n\n    Args:\n      op: The tf op to benchmark.\n      iters: Number of times to repeat the timing.\n      warmup: If true, warms up the session caches by running an untimed run.\n      session_config: A ConfigProto protocol buffer with configuration options\n        for the session. Applicable only for benchmarking in graph mode.\n\n    Returns:\n      A float, representing the per-execution wall time of the op in seconds.\n      This is the median time (with respect to `iters`) it takes for the op\n      to be executed `iters` num of times.\n    '
        if context.executing_eagerly():
            return self._run_eager_benchmark(iterable=op, iters=iters, warmup=warmup)
        return self._run_graph_benchmark(iterable=op, iters=iters, warmup=warmup, session_config=session_config)

    def run_benchmark(self, dataset, num_elements, iters=1, warmup=True, apply_default_optimizations=False, session_config=None):
        if False:
            return 10
        'Benchmarks the dataset.\n\n    Runs the dataset `iters` times. In each iteration, the benchmark measures\n    the time it takes to go through `num_elements` elements of the dataset.\n\n    Args:\n      dataset: Dataset to benchmark.\n      num_elements: Number of dataset elements to iterate through each benchmark\n        iteration.\n      iters: Number of times to repeat the timing.\n      warmup: If true, warms up the session caches by running an untimed run.\n      apply_default_optimizations: Determines whether default optimizations\n        should be applied.\n      session_config: A ConfigProto protocol buffer with configuration options\n        for the session. Applicable only for benchmarking in graph mode.\n\n    Returns:\n      A float, representing the per-element wall time of the dataset in seconds.\n      This is the median time (with respect to `iters`) it takes for the dataset\n      to go through `num_elements` elements, divided by `num_elements.`\n    '
        options = options_lib.Options()
        options.experimental_optimization.apply_default_optimizations = apply_default_optimizations
        dataset = dataset.with_options(options)
        dataset = dataset.skip(num_elements - 1)
        if context.executing_eagerly():
            median_duration = self._run_eager_benchmark(iterable=dataset, iters=iters, warmup=warmup)
            return median_duration / float(num_elements)
        iterator = dataset_ops.make_initializable_iterator(dataset)
        next_element = iterator.get_next()
        op = nest.flatten(next_element)[0].op
        median_duration = self._run_graph_benchmark(iterable=op, iters=iters, warmup=warmup, session_config=session_config, initializer=iterator.initializer)
        return median_duration / float(num_elements)

    def run_and_report_benchmark(self, dataset, num_elements, name, iters=5, extras=None, warmup=True, apply_default_optimizations=False, session_config=None):
        if False:
            while True:
                i = 10
        'Benchmarks the dataset and reports the stats.\n\n    Runs the dataset `iters` times. In each iteration, the benchmark measures\n    the time it takes to go through `num_elements` elements of the dataset.\n    This is followed by logging/printing the benchmark stats.\n\n    Args:\n      dataset: Dataset to benchmark.\n      num_elements: Number of dataset elements to iterate through each benchmark\n        iteration.\n      name: Name of the benchmark.\n      iters: Number of times to repeat the timing.\n      extras: A dict which maps string keys to additional benchmark info.\n      warmup: If true, warms up the session caches by running an untimed run.\n      apply_default_optimizations: Determines whether default optimizations\n        should be applied.\n      session_config: A ConfigProto protocol buffer with configuration options\n        for the session. Applicable only for benchmarking in graph mode.\n\n    Returns:\n      A float, representing the per-element wall time of the dataset in seconds.\n      This is the median time (with respect to `iters`) it takes for the dataset\n      to go through `num_elements` elements, divided by `num_elements.`\n    '
        wall_time = self.run_benchmark(dataset=dataset, num_elements=num_elements, iters=iters, warmup=warmup, apply_default_optimizations=apply_default_optimizations, session_config=session_config)
        if extras is None:
            extras = {}
        if context.executing_eagerly():
            name = '{}.eager'.format(name)
            extras['implementation'] = 'eager'
        else:
            name = '{}.graph'.format(name)
            extras['implementation'] = 'graph'
        extras['num_elements'] = num_elements
        self.report_benchmark(wall_time=wall_time, iters=iters, name=name, extras=extras)
        return wall_time