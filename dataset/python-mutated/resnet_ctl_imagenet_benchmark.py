"""Executes CTL benchmarks and accuracy tests."""
from __future__ import print_function
import os
import time
from absl import flags
import tensorflow as tf
from official.vision.image_classification import common
from official.vision.image_classification import resnet_ctl_imagenet_main
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark
from official.utils.flags import core as flags_core
MIN_TOP_1_ACCURACY = 0.76
MAX_TOP_1_ACCURACY = 0.77
FLAGS = flags.FLAGS

class CtlBenchmark(PerfZeroBenchmark):
    """Base benchmark class with methods to simplify testing."""

    def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
        if False:
            print('Hello World!')
        self.output_dir = output_dir
        self.default_flags = default_flags or {}
        self.flag_methods = flag_methods or {}
        super(CtlBenchmark, self).__init__(output_dir=self.output_dir, default_flags=self.default_flags, flag_methods=self.flag_methods)

    def _report_benchmark(self, stats, wall_time_sec, top_1_max=None, top_1_min=None, total_batch_size=None, log_steps=None, warmup=1):
        if False:
            print('Hello World!')
        "Report benchmark results by writing to local protobuf file.\n\n    Args:\n      stats: dict returned from keras models with known entries.\n      wall_time_sec: the during of the benchmark execution in seconds\n      top_1_max: highest passing level for top_1 accuracy.\n      top_1_min: lowest passing level for top_1 accuracy.\n      total_batch_size: Global batch-size.\n      log_steps: How often the log was created for stats['step_timestamp_log'].\n      warmup: number of entries in stats['step_timestamp_log'] to ignore.\n    "
        metrics = []
        if 'eval_acc' in stats:
            metrics.append({'name': 'accuracy_top_1', 'value': stats['eval_acc'], 'min_value': top_1_min, 'max_value': top_1_max})
            metrics.append({'name': 'eval_loss', 'value': stats['eval_loss']})
            metrics.append({'name': 'top_1_train_accuracy', 'value': stats['train_acc']})
            metrics.append({'name': 'train_loss', 'value': stats['train_loss']})
        if warmup and 'step_timestamp_log' in stats and (len(stats['step_timestamp_log']) > warmup):
            time_log = stats['step_timestamp_log']
            elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
            num_examples = total_batch_size * log_steps * (len(time_log) - warmup - 1)
            examples_per_sec = num_examples / elapsed
            metrics.append({'name': 'exp_per_second', 'value': examples_per_sec})
        if 'avg_exp_per_second' in stats:
            metrics.append({'name': 'avg_exp_per_second', 'value': stats['avg_exp_per_second']})
        flags_str = flags_core.get_nondefault_flags_as_str()
        self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics, extras={'flags': flags_str})

class Resnet50CtlAccuracy(CtlBenchmark):
    """Benchmark accuracy tests for ResNet50 in CTL."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            while True:
                i = 10
        'A benchmark class.\n\n    Args:\n      output_dir: directory where to output e.g. log files\n      root_data_dir: directory under which to look for dataset\n      **kwargs: arbitrary named arguments. This is needed to make the\n        constructor forward compatible in case PerfZero provides more named\n        arguments before updating the constructor.\n    '
        flag_methods = [common.define_keras_flags]
        self.data_dir = os.path.join(root_data_dir, 'imagenet')
        super(Resnet50CtlAccuracy, self).__init__(output_dir=output_dir, flag_methods=flag_methods)

    def benchmark_8_gpu(self):
        if False:
            while True:
                i = 10
        'Test Keras model with eager, dist_strat and 8 GPUs.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.data_dir = self.data_dir
        FLAGS.batch_size = 128 * 8
        FLAGS.train_epochs = 90
        FLAGS.epochs_between_evals = 10
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
        FLAGS.dtype = 'fp32'
        FLAGS.datasets_num_private_threads = 14
        self._run_and_report_benchmark()

    def benchmark_8_gpu_fp16(self):
        if False:
            print('Hello World!')
        'Test Keras model with eager, 8 GPUs with tf.keras mixed precision.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.data_dir = self.data_dir
        FLAGS.batch_size = 256 * 8
        FLAGS.train_epochs = 90
        FLAGS.epochs_between_evals = 10
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_fp16')
        FLAGS.dtype = 'fp16'
        FLAGS.datasets_num_private_threads = 14
        self._run_and_report_benchmark()

    def benchmark_8_gpu_amp(self):
        if False:
            i = 10
            return i + 15
        'Test Keras model with 8 GPUs and mixed precision via graph rewrite.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.data_dir = self.data_dir
        FLAGS.batch_size = 256 * 8
        FLAGS.train_epochs = 90
        FLAGS.epochs_between_evals = 10
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp')
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        FLAGS.datasets_num_private_threads = 14
        self._run_and_report_benchmark()

    def _run_and_report_benchmark(self):
        if False:
            for i in range(10):
                print('nop')
        start_time_sec = time.time()
        stats = resnet_ctl_imagenet_main.run(flags.FLAGS)
        wall_time_sec = time.time() - start_time_sec
        super(Resnet50CtlAccuracy, self)._report_benchmark(stats, wall_time_sec, top_1_min=MIN_TOP_1_ACCURACY, top_1_max=MAX_TOP_1_ACCURACY, total_batch_size=FLAGS.batch_size, log_steps=100)

    def _get_model_dir(self, folder_name):
        if False:
            i = 10
            return i + 15
        return os.path.join(self.output_dir, folder_name)

class Resnet50CtlBenchmarkBase(CtlBenchmark):
    """Resnet50 benchmarks."""

    def __init__(self, output_dir=None, default_flags=None):
        if False:
            while True:
                i = 10
        flag_methods = [common.define_keras_flags]
        super(Resnet50CtlBenchmarkBase, self).__init__(output_dir=output_dir, flag_methods=flag_methods, default_flags=default_flags)

    def _run_and_report_benchmark(self):
        if False:
            while True:
                i = 10
        start_time_sec = time.time()
        stats = resnet_ctl_imagenet_main.run(FLAGS)
        wall_time_sec = time.time() - start_time_sec
        warmup = (FLAGS.train_steps - 100) // FLAGS.log_steps
        super(Resnet50CtlBenchmarkBase, self)._report_benchmark(stats, wall_time_sec, total_batch_size=FLAGS.batch_size, log_steps=FLAGS.log_steps, warmup=warmup)

    def benchmark_1_gpu_no_dist_strat(self):
        if False:
            return 10
        'Test Keras model with 1 GPU, no distribution strategy.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.distribution_strategy = 'off'
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_no_dist_strat')
        FLAGS.batch_size = 128
        self._run_and_report_benchmark()

    def benchmark_1_gpu(self):
        if False:
            while True:
                i = 10
        'Test Keras model with 1 GPU.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.distribution_strategy = 'default'
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu')
        FLAGS.batch_size = 128
        self._run_and_report_benchmark()

    def benchmark_1_gpu_amp(self):
        if False:
            for i in range(10):
                print('nop')
        'Test Keras model with 1 GPU with automatic mixed precision.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.distribution_strategy = 'default'
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_amp')
        FLAGS.batch_size = 256
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu_amp(self):
        if False:
            while True:
                i = 10
        'Test Keras model with XLA and 1 GPU with automatic mixed precision.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.distribution_strategy = 'default'
        FLAGS.model_dir = self._get_model_dir('benchmark_xla_1_gpu_amp')
        FLAGS.batch_size = 256
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def benchmark_1_gpu_eager(self):
        if False:
            print('Hello World!')
        'Test Keras model with 1 GPU in pure eager mode.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.distribution_strategy = 'default'
        FLAGS.model_dir = self._get_model_dir('benchmark_1_gpu_eager')
        FLAGS.batch_size = 64
        FLAGS.use_tf_function = False
        FLAGS.single_l2_loss_op = True
        self._run_and_report_benchmark()

    def benchmark_8_gpu(self):
        if False:
            i = 10
            return i + 15
        'Test Keras model with 8 GPUs.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.distribution_strategy = 'default'
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu')
        FLAGS.batch_size = 128 * 8
        self._run_and_report_benchmark()

    def benchmark_8_gpu_amp(self):
        if False:
            i = 10
            return i + 15
        'Test Keras model with 8 GPUs with automatic mixed precision.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.distribution_strategy = 'default'
        FLAGS.model_dir = self._get_model_dir('benchmark_8_gpu_amp')
        FLAGS.batch_size = 256 * 8
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        self._run_and_report_benchmark()

    def benchmark_xla_8_gpu_amp(self):
        if False:
            print('Hello World!')
        'Test Keras model with XLA and 8 GPUs with automatic mixed precision.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.distribution_strategy = 'default'
        FLAGS.model_dir = self._get_model_dir('benchmark_xla_8_gpu_amp')
        FLAGS.batch_size = 256 * 8
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def fill_report_object(self, stats):
        if False:
            print('Hello World!')
        super(Resnet50CtlBenchmarkBase, self).fill_report_object(stats, total_batch_size=FLAGS.batch_size, log_steps=FLAGS.log_steps)

class Resnet50CtlBenchmarkSynth(Resnet50CtlBenchmarkBase):
    """Resnet50 synthetic benchmark tests."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            while True:
                i = 10
        def_flags = {}
        def_flags['skip_eval'] = True
        def_flags['use_synthetic_data'] = True
        def_flags['train_steps'] = 110
        def_flags['log_steps'] = 10
        super(Resnet50CtlBenchmarkSynth, self).__init__(output_dir=output_dir, default_flags=def_flags)

class Resnet50CtlBenchmarkReal(Resnet50CtlBenchmarkBase):
    """Resnet50 real data benchmark tests."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            print('Hello World!')
        def_flags = {}
        def_flags['skip_eval'] = True
        def_flags['data_dir'] = os.path.join(root_data_dir, 'imagenet')
        def_flags['train_steps'] = 110
        def_flags['log_steps'] = 10
        super(Resnet50CtlBenchmarkReal, self).__init__(output_dir=output_dir, default_flags=def_flags)
if __name__ == '__main__':
    tf.test.main()