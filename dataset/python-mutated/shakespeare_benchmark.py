"""Executes Shakespeare (LSTM) benchmark and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from absl import flags
import tensorflow as tf
from official.staging.shakespeare import shakespeare_main
from official.utils.flags import core as flags_core
from official.utils.misc import keras_utils
from official.utils.testing.perfzero_benchmark import PerfZeroBenchmark
SHAKESPEARE_TRAIN_DATA = 'shakespeare/shakespeare.txt'
TMP_DIR = os.getenv('TMPDIR')
FLAGS = flags.FLAGS

class ShakespeareBenchmarkBase(PerfZeroBenchmark):
    """Base class for Shakespeare (LSTM) benchmark and accuracy tests."""

    def __init__(self, output_dir=None, default_flags=None, root_data_dir=None):
        if False:
            return 10
        super(ShakespeareBenchmarkBase, self).__init__(output_dir=output_dir, default_flags=default_flags, flag_methods=[shakespeare_main.define_flags])

    def _run_and_report_benchmark(self, top_1_train_min=0.91, top_1_train_max=0.94, warmup=1, log_steps=100):
        if False:
            return 10
        'Report benchmark results by writing to local protobuf file.\n\n    Average epoch time is calculated by skipping the first epoch. This average\n    ignores time spent between epoch and is recorded by begin and end epoch. To\n    skip accuracy check set `top_1_train_min=None`.\n\n    Args:\n      top_1_train_min: lowest passing value.\n      top_1_train_max: highest passing value.\n      warmup: number of entries in `timestamp_log` to ignore.\n      log_steps: How often the log was created for `timestamp_log`.\n    '
        total_batch_size = FLAGS.batch_size
        metrics = []
        start_time_sec = time.time()
        stats = shakespeare_main.run(FLAGS)
        wall_time_sec = time.time() - start_time_sec
        if top_1_train_min:
            metrics.append({'name': 'accuracy_top_1_train', 'value': stats['history']['RecallAt1'][-1], 'min_value': top_1_train_min, 'max_value': top_1_train_max})
        for callback in stats['callbacks']:
            if isinstance(callback, keras_utils.TimeHistory):
                epoch_timings = callback.epoch_runtime_log
                average_time = sum(epoch_timings[1:]) / len(epoch_timings[1:])
                metrics.append({'name': 'avg_epoch_time', 'value': average_time})
            time_log = callback.timestamp_log
            elapsed = time_log[-1].timestamp - time_log[warmup].timestamp
            num_examples = total_batch_size * log_steps * (len(time_log) - warmup - 1)
            examples_per_sec = num_examples / elapsed
            metrics.append({'name': 'exp_per_second', 'value': examples_per_sec})
        flags_str = flags_core.get_nondefault_flags_as_str()
        self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics, extras={'flags': flags_str})

class ShakespeareAccuracy(ShakespeareBenchmarkBase):
    """Shakespeare accuracy tests.

  This is not an ideal test. The best we can use for the accuracy check is to
  validate top_1 of the training set. At batch size 64 the top_1 training
  stabilizes to ~0.92 around 40-45 epochs.
  """

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            print('Hello World!')
        'Shakespeare accuracy tests.\n\n    Args:\n      output_dir: directory where to output e.g. log files\n      root_data_dir: directory under which to look for dataset\n      **kwargs: arbitrary named arguments. This is needed to make the\n                constructor forward compatible in case PerfZero provides more\n                named arguments before updating the constructor.\n    '
        self.train_data = os.path.join(root_data_dir, SHAKESPEARE_TRAIN_DATA)
        super(ShakespeareAccuracy, self).__init__(output_dir=output_dir, root_data_dir=root_data_dir)

    def benchmark_cpu(self):
        if False:
            for i in range(10):
                print('nop')
        'Benchmark cpu.'
        self._setup()
        FLAGS.num_gpus = 0
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        self._run_and_report_benchmark()

    def benchmark_cpu_no_ds_run_eagerly(self):
        if False:
            return 10
        'Benchmark cpu without distribution strategies and run eagerly.'
        self._setup()
        FLAGS.num_gpus = 0
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        FLAGS.run_eagerly = True
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_1_gpu(self):
        if False:
            print('Hello World!')
        'Benchmark 1 gpu.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_ds(self):
        if False:
            return 10
        'Benchmark 1 gpu without distribution strategies.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_ds_run_eagerly(self):
        if False:
            print('Hello World!')
        'Benchmark 1 gpu without distribution strategies and run eagerly.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        FLAGS.run_eagerly = True
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_ds_force_v2(self):
        if False:
            return 10
        'Benchmark 1 gpu no ds with force_v2 in keras.compile.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        FLAGS.force_v2_in_keras_compile = True
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark 1 gpu w/xla.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def benchmark_8_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        'Benchmark 8 gpu.\n\n    This is test is for accuracy not scaling.  The batch-size is not scaled to\n    the number of gpus.\n    '
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.training_data = self.train_data
        FLAGS.batch_size = 64
        FLAGS.train_epochs = 43
        FLAGS.model_dir = ''
        self._run_and_report_benchmark()

class ShakespeareKerasBenchmarkReal(ShakespeareBenchmarkBase):
    """Benchmark accuracy tests."""

    def __init__(self, output_dir=None, root_data_dir=TMP_DIR, **kwargs):
        if False:
            print('Hello World!')
        'Benchmark tests w/Keras.\n\n    Args:\n      output_dir: directory where to output e.g. log files\n      root_data_dir: directory under which to look for dataset\n      **kwargs: arbitrary named arguments. This is needed to make the\n                constructor forward compatible in case PerfZero provides more\n                named arguments before updating the constructor.\n    '
        self.train_data = os.path.join(root_data_dir, SHAKESPEARE_TRAIN_DATA)
        def_flags = {}
        def_flags['training_data'] = self.train_data
        def_flags['model_dir'] = ''
        def_flags['train_epochs'] = 4
        def_flags['log_steps'] = 50
        super(ShakespeareKerasBenchmarkReal, self).__init__(output_dir=output_dir, root_data_dir=root_data_dir, default_flags=def_flags)

    def benchmark_cpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark cpu.'
        self._setup()
        FLAGS.num_gpus = 0
        FLAGS.batch_size = 64
        self._run_and_report_benchmark()

    def benchmark_cpu_no_ds_run_eagerly(self):
        if False:
            i = 10
            return i + 15
        'Benchmark cpu without distribution strategy and run eagerly.'
        self._setup()
        FLAGS.num_gpus = 0
        FLAGS.batch_size = 64
        FLAGS.distribution_strategy = 'off'
        FLAGS.run_eagerly = True
        self._run_and_report_benchmark()

    def benchmark_cpu_no_ds(self):
        if False:
            return 10
        'Benchmark cpu without distribution strategy.'
        self._setup()
        FLAGS.num_gpus = 0
        FLAGS.batch_size = 64
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_cpu_no_ds_force_v2(self):
        if False:
            return 10
        'Benchmark cpu no ds, and force v2.'
        self._setup()
        FLAGS.num_gpus = 0
        FLAGS.batch_size = 64
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_1_gpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark 1 gpu.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_cudnn(self):
        if False:
            return 10
        'Benchmark 1 gpu with CuDNN disabled.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64
        FLAGS.cudnn = False
        FLAGS.enable_eager = keras_utils.is_v2_0()
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_ds(self):
        if False:
            return 10
        'Benchmark 1 gpu without distribution strategies.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_ds_force_v2(self):
        if False:
            return 10
        'Benchmark 1 gpu no ds, and force v2.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64
        FLAGS.force_v2_in_keras_compile = True
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_ds_run_eagerly(self):
        if False:
            i = 10
            return i + 15
        'Benchmark 1 gpu.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64
        FLAGS.run_eagerly = True
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark 1 gpu.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu_no_cudnn(self):
        if False:
            print('Hello World!')
        'Benchmark 1 gpu w/xla and CuDNN disabled.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64
        FLAGS.cudnn = False
        FLAGS.enable_eager = keras_utils.is_v2_0()
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def benchmark_8_gpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark 8 gpu.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.batch_size = 64 * 8
        FLAGS.log_steps = 10
        self._run_and_report_benchmark()

    def benchmark_8_gpu_no_cudnn(self):
        if False:
            i = 10
            return i + 15
        'Benchmark 8 gpu with CuDNN disabled.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.batch_size = 64 * 8
        FLAGS.log_steps = 10
        FLAGS.cudnn = False
        FLAGS.enable_eager = keras_utils.is_v2_0()
        self._run_and_report_benchmark()

    def benchmark_xla_8_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        'Benchmark 8 gpu w/xla.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = 64 * 8
        FLAGS.log_steps = 10
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def benchmark_xla_8_gpu_no_cudnn(self):
        if False:
            while True:
                i = 10
        'Benchmark 8 gpu w/xla and CuDNN disabled.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.batch_size = 64 * 8
        FLAGS.log_steps = 10
        FLAGS.cudnn = False
        FLAGS.enable_eager = keras_utils.is_v2_0()
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def _run_and_report_benchmark(self):
        if False:
            return 10
        'Run and report benchmark.'
        super(ShakespeareKerasBenchmarkReal, self)._run_and_report_benchmark(top_1_train_min=None, log_steps=FLAGS.log_steps)
if __name__ == '__main__':
    tf.test.main()