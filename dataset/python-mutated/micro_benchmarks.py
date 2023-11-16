"""Micro benchmark.

bazel run -c opt --config=cuda \\
  //third_party/tensorflow/python/ops/numpy_ops/integration_test/benchmarks:micro_benchmarks -- \\
  --number=100 --repeat=100 \\
  --benchmark_filter=.
"""
import gc
import time
from absl import flags
from absl import logging
import numpy as np
import tensorflow.compat.v2 as tf
from tensorflow.python.ops.numpy_ops.integration_test.benchmarks import numpy_mlp
from tensorflow.python.ops.numpy_ops.integration_test.benchmarks import tf_numpy_mlp
FLAGS = flags.FLAGS
tfnp = tf.experimental.numpy
flags.DEFINE_integer('repeat', 100, '#Measurements per benchmark.')
flags.DEFINE_integer('number', 100, '#Runs per a measure.')

class MicroBenchmarks(tf.test.Benchmark):
    """Main micro benchmark class."""

    def _benchmark_and_report(self, name, fn, repeat=None, number=None):
        if False:
            i = 10
            return i + 15
        'Run fn repeat * number times, report time, and return fastest time.'
        repeat = repeat or int(FLAGS.repeat)
        number = number or int(FLAGS.number)
        fn()
        times = []
        for _ in range(repeat):
            gc.disable()
            start = time.time()
            for _ in range(number):
                fn()
            times.append(time.time() - start)
            gc.enable()
            gc.collect()
        fastest_time_us = min(times) * 1000000.0 / number
        total_time = sum(times)
        self.report_benchmark(name=name, wall_time=total_time, extras={'fastest_time_us': fastest_time_us})
        return fastest_time_us

    def benchmark_tf_np_mlp_inference_batch_1_cpu(self):
        if False:
            i = 10
            return i + 15
        with tf.device('/CPU:0'):
            model = tf_numpy_mlp.MLP()
            x = tfnp.ones(shape=(1, 10)).astype(np.float32)
            self._benchmark_and_report(self._get_name(), lambda : model.inference(x))

    def benchmark_tf_np_tf_function_mlp_inference_batch_1_cpu(self):
        if False:
            return 10
        with tf.device('/CPU:0'):
            model = tf_numpy_mlp.MLP()
            x = tfnp.ones(shape=(1, 10)).astype(np.float32)
            self._benchmark_and_report(self._get_name(), tf.function(lambda : model.inference(x)))

    def benchmark_numpy_mlp_inference_batch_1_cpu(self):
        if False:
            while True:
                i = 10
        model = numpy_mlp.MLP()
        x = np.random.uniform(size=(1, 10)).astype(np.float32, copy=False)
        self._benchmark_and_report(self._get_name(), lambda : model.inference(x))

    def _benchmark_np_and_tf_np(self, name, op, args, repeat=None):
        if False:
            while True:
                i = 10
        fn = getattr(np, op)
        assert fn is not None
        np_time = self._benchmark_and_report('{}_numpy'.format(name), lambda : fn(*args), repeat=repeat)
        fn = getattr(tfnp, op)
        assert fn is not None
        with tf.device('CPU:0'):
            tf_time = self._benchmark_and_report('{}_tfnp_cpu'.format(name), lambda : fn(*args), repeat=repeat)
        return (np_time, tf_time)

    def _print_times(self, op, sizes, times):
        if False:
            print('Hello World!')
        print('For np.{}:'.format(op))
        print('{:<15}  {:>11}  {:>11}'.format('Size', 'NP time', 'TF NP Time'))
        for (size, (np_time, tf_time)) in zip(sizes, times):
            print('{:<15} {:>10.5}us {:>10.5}us'.format(str(size), np_time, tf_time))
        print()

    def _benchmark_np_and_tf_np_unary(self, op):
        if False:
            return 10
        sizes = [(100,), (10000,), (1000000,)]
        repeats = [FLAGS.repeat] * 2 + [10]
        times = []
        for (size, repeat) in zip(sizes, repeats):
            x = np.random.uniform(size=size).astype(np.float32, copy=False)
            name = '{}_{}'.format(self._get_name(), size)
            times.append(self._benchmark_np_and_tf_np(name, op, (x,), repeat))
        self._print_times(op, sizes, times)

    def benchmark_count_nonzero(self):
        if False:
            print('Hello World!')
        self._benchmark_np_and_tf_np_unary('count_nonzero')

    def benchmark_log(self):
        if False:
            return 10
        self._benchmark_np_and_tf_np_unary('log')

    def benchmark_exp(self):
        if False:
            while True:
                i = 10
        self._benchmark_np_and_tf_np_unary('exp')

    def benchmark_tanh(self):
        if False:
            print('Hello World!')
        self._benchmark_np_and_tf_np_unary('tanh')

    def benchmark_matmul(self):
        if False:
            return 10
        sizes = [(2, 2), (10, 10), (100, 100), (200, 200), (1000, 1000)]
        repeats = [FLAGS.repeat] * 3 + [50, 10]
        times = []
        for (size, repeat) in zip(sizes, repeats):
            x = np.random.uniform(size=size).astype(np.float32, copy=False)
            name = '{}_{}'.format(self._get_name(), size)
            times.append(self._benchmark_np_and_tf_np(name, 'matmul', (x, x), repeat=repeat))
        self._print_times('matmul', sizes, times)
if __name__ == '__main__':
    logging.set_verbosity(logging.WARNING)
    tf.enable_v2_behavior()
    tf.test.main()