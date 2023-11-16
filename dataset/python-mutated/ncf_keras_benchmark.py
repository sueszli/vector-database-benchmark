"""Executes Keras benchmarks and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
from official.recommendation import ncf_common
from official.recommendation import ncf_keras_main
from official.utils.flags import core
FLAGS = flags.FLAGS
NCF_DATA_DIR_NAME = 'movielens_data'
NCF_TF_DATA_1M_BATCH_DIR_NAME = 'gs://tf-perfzero-data/movielens_data/ncf_8gpu_1M_batch'

class NCFKerasBenchmarkBase(tf.test.Benchmark):
    """Base class for NCF model benchmark."""
    local_flags = None

    def __init__(self, output_dir=None, default_flags=None, **kwargs):
        if False:
            i = 10
            return i + 15
        self.output_dir = output_dir
        self.default_flags = default_flags or {}

    def _setup(self):
        if False:
            print('Hello World!')
        'Sets up and resets flags before each test.'
        assert tf.version.VERSION.startswith('2.')
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        if NCFKerasBenchmarkBase.local_flags is None:
            ncf_common.define_ncf_flags()
            flags.FLAGS(['foo'])
            core.set_defaults(**self.default_flags)
            saved_flag_values = flagsaver.save_flag_values()
            NCFKerasBenchmarkBase.local_flags = saved_flag_values
        else:
            flagsaver.restore_flag_values(NCFKerasBenchmarkBase.local_flags)

    def _run_and_report_benchmark(self, hr_at_10_min=0, hr_at_10_max=0):
        if False:
            for i in range(10):
                print('nop')
        start_time_sec = time.time()
        stats = ncf_keras_main.run_ncf(FLAGS)
        wall_time_sec = time.time() - start_time_sec
        metrics = []
        metrics.append({'name': 'exp_per_second', 'value': stats['avg_exp_per_second']})
        if hr_at_10_min > 0:
            metrics.append({'name': 'hr_at_10', 'value': stats['eval_hit_rate'], 'min_value': hr_at_10_min, 'max_value': hr_at_10_max})
            metrics.append({'name': 'train_loss', 'value': stats['loss']})
        self.report_benchmark(iters=-1, wall_time=wall_time_sec, metrics=metrics)

class NCFKerasAccuracy(NCFKerasBenchmarkBase):
    """Benchmark NCF model using real data."""

    def __init__(self, output_dir=None, root_data_dir=None, default_flags=None, **kwargs):
        if False:
            while True:
                i = 10
        root_data_dir = root_data_dir if root_data_dir else ''
        default_flags = {}
        default_flags['dataset'] = 'ml-20m'
        default_flags['num_gpus'] = 1
        default_flags['train_epochs'] = 10
        default_flags['clean'] = True
        default_flags['batch_size'] = 99000
        default_flags['learning_rate'] = 0.00382059
        default_flags['beta1'] = 0.783529
        default_flags['beta2'] = 0.909003
        default_flags['epsilon'] = 1.45439e-07
        default_flags['layers'] = [256, 256, 128, 64]
        default_flags['num_factors'] = 64
        default_flags['hr_threshold'] = 0.635
        default_flags['ml_perf'] = True
        default_flags['use_synthetic_data'] = False
        default_flags['data_dir'] = os.path.join(root_data_dir, NCF_DATA_DIR_NAME)
        super(NCFKerasAccuracy, self).__init__(output_dir=output_dir, default_flags=default_flags, **kwargs)

    def _run_and_report_benchmark_mlperf_like(self):
        if False:
            return 10
        'Run test and report results.\n\n    Note: MLPerf like tests are not tuned to hit a specific hr@10 value, but\n    we want it recorded.\n    '
        self._run_and_report_benchmark(hr_at_10_min=0.61)

    def _run_and_report_benchmark(self, hr_at_10_min=0.63, hr_at_10_max=0.645):
        if False:
            print('Hello World!')
        'Run test and report results.\n\n    Note: Target is 0.635, but some runs are below that level. Until we have\n    multi-run tests, we have to accept a lower target.\n\n    Args:\n      hr_at_10_min: Minimum acceptable hr@10 value.\n      hr_at_10_max: Maximum acceptable hr@10 value.\n    '
        super(NCFKerasAccuracy, self)._run_and_report_benchmark(hr_at_10_min=hr_at_10_min, hr_at_10_max=hr_at_10_max)

    def benchmark_1_gpu_early_stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._setup()
        FLAGS.early_stopping = True
        self._run_and_report_benchmark()

    def benchmark_1_gpu_force_v1_path_early_stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._setup()
        FLAGS.early_stopping = True
        FLAGS.force_v2_in_keras_compile = False
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_dist_strat_early_stop(self):
        if False:
            return 10
        self._setup()
        FLAGS.distribution_strategy = 'off'
        FLAGS.early_stopping = True
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_dist_strat_force_v1_path_early_stop(self):
        if False:
            return 10
        self._setup()
        FLAGS.distribution_strategy = 'off'
        FLAGS.early_stopping = True
        FLAGS.force_v2_in_keras_compile = False
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_dist_strat_run_eagerly_early_stop(self):
        if False:
            i = 10
            return i + 15
        self._setup()
        FLAGS.distribution_strategy = 'off'
        FLAGS.early_stopping = True
        FLAGS.run_eagerly = True
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu_early_stop(self):
        if False:
            for i in range(10):
                print('nop')
        self._setup()
        FLAGS.early_stopping = True
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu_force_v1_path_early_stop(self):
        if False:
            return 10
        self._setup()
        FLAGS.early_stopping = True
        FLAGS.enable_xla = True
        FLAGS.force_v2_in_keras_compile = False
        self._run_and_report_benchmark()

    def benchmark_1_gpu_ctl_early_stop(self):
        if False:
            print('Hello World!')
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.early_stopping = True
        self._run_and_report_benchmark()

    def benchmark_1_gpu_ctl_run_eagerly_early_stop(self):
        if False:
            while True:
                i = 10
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.early_stopping = True
        FLAGS.run_eagerly = True
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu_ctl_early_stop(self):
        if False:
            while True:
                i = 10
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.early_stopping = True
        FLAGS.enable_xla = True
        self._run_and_report_benchmark()

    def benchmark_2_gpus_early_stop(self):
        if False:
            print('Hello World!')
        self._setup()
        FLAGS.early_stopping = True
        FLAGS.num_gpus = 2
        FLAGS.eval_batch_size = 160000
        self._run_and_report_benchmark()

    def benchmark_2_gpus_ctl_early_stop(self):
        if False:
            return 10
        'NCF with custom training loop. Works only in TF 2.0.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.early_stopping = True
        FLAGS.num_gpus = 2
        FLAGS.eval_batch_size = 160000
        self._run_and_report_benchmark()

    def benchmark_1_gpu_mlperf_like(self):
        if False:
            for i in range(10):
                print('nop')
        '1 GPU using keras fit/compile.'
        self._setup()
        FLAGS.train_epochs = 7
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_no_dist_strat_force_v1_path_mlperf_like(self):
        if False:
            print('Hello World!')
        '1 GPU using compile/fit without dist_strat.'
        self._setup()
        FLAGS.train_epochs = 7
        FLAGS.distribution_strategy = 'off'
        FLAGS.force_v2_in_keras_compile = False
        self._run_and_report_benchmark()

    def benchmark_1_gpu_no_dist_strat_mlperf_like(self):
        if False:
            for i in range(10):
                print('nop')
        '1 GPU using compile/fit without dist_strat.'
        self._setup()
        FLAGS.train_epochs = 7
        FLAGS.distribution_strategy = 'off'
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_no_dist_strat_run_eagerly_mlperf_like(self):
        if False:
            return 10
        self._setup()
        FLAGS.train_epochs = 7
        FLAGS.distribution_strategy = 'off'
        FLAGS.run_eagerly = True
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_xla_1_gpu_mlperf_like(self):
        if False:
            i = 10
            return i + 15
        '1 GPU using compile/fit with XLA.'
        self._setup()
        FLAGS.train_epochs = 7
        FLAGS.enable_xla = True
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_ctl_mlperf_like(self):
        if False:
            print('Hello World!')
        '1 GPU using CTL.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.train_epochs = 7
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_ctl_fp16_mlperf_like(self):
        if False:
            return 10
        '1 GPU using CTL and FP16.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.train_epochs = 7
        FLAGS.dtype = 'fp16'
        FLAGS.loss_scale = 8192
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_fp16_mlperf_like(self):
        if False:
            i = 10
            return i + 15
        '1 GPU using FP16.'
        self._setup()
        FLAGS.train_epochs = 7
        FLAGS.dtype = 'fp16'
        FLAGS.loss_scale = 8192
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_ctl_fp16_graph_rewrite_mlperf_like(self):
        if False:
            i = 10
            return i + 15
        '1 GPU using CTL and FP16 graph rewrite.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.train_epochs = 7
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        FLAGS.loss_scale = 8192
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_fp16_graph_rewrite_mlperf_like(self):
        if False:
            while True:
                i = 10
        '1 GPU using FP16 graph rewrite.'
        self._setup()
        FLAGS.train_epochs = 7
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        FLAGS.loss_scale = 8192
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_1_gpu_ctl_run_eagerly_mlperf_like(self):
        if False:
            for i in range(10):
                print('nop')
        '1 GPU using CTL with eager and distribution strategy.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.run_eagerly = True
        FLAGS.train_epochs = 7
        self._run_and_report_benchmark()

    def benchmark_xla_1_gpu_ctl_mlperf_like(self):
        if False:
            for i in range(10):
                print('nop')
        '1 GPU using CTL with XLA.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.enable_xla = True
        FLAGS.train_epochs = 7
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_xla_1_gpu_fp16_mlperf_like(self):
        if False:
            i = 10
            return i + 15
        '1 GPU using with XLA and FP16.'
        self._setup()
        FLAGS.enable_xla = True
        FLAGS.train_epochs = 7
        FLAGS.dtype = 'fp16'
        FLAGS.loss_scale = 8192
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_xla_1_gpu_ctl_fp16_mlperf_like(self):
        if False:
            print('Hello World!')
        '1 GPU using CTL with XLA and FP16.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.enable_xla = True
        FLAGS.train_epochs = 7
        FLAGS.dtype = 'fp16'
        FLAGS.loss_scale = 8192
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_8_gpu_mlperf_like(self):
        if False:
            i = 10
            return i + 15
        '8 GPU using keras fit/compile.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.train_epochs = 17
        FLAGS.batch_size = 1048576
        FLAGS.eval_batch_size = 160000
        FLAGS.learning_rate = 0.0045
        FLAGS.beta1 = 0.25
        FLAGS.beta2 = 0.5
        FLAGS.epsilon = 1e-08
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_8_gpu_force_v1_path_mlperf_like(self):
        if False:
            for i in range(10):
                print('nop')
        '8 GPU using keras fit/compile v1 codepath.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.train_epochs = 17
        FLAGS.batch_size = 1048576
        FLAGS.eval_batch_size = 160000
        FLAGS.learning_rate = 0.0045
        FLAGS.beta1 = 0.25
        FLAGS.beta2 = 0.5
        FLAGS.epsilon = 1e-08
        FLAGS.force_v2_in_keras_compile = False
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_8_gpu_ctl_mlperf_like(self):
        if False:
            for i in range(10):
                print('nop')
        '8 GPU using CTL.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.num_gpus = 8
        FLAGS.train_epochs = 17
        FLAGS.batch_size = 1048576
        FLAGS.eval_batch_size = 160000
        FLAGS.learning_rate = 0.0045
        FLAGS.beta1 = 0.25
        FLAGS.beta2 = 0.5
        FLAGS.epsilon = 1e-08
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_8_gpu_tf_data_ctl_mlperf_like(self):
        if False:
            while True:
                i = 10
        '8 GPU using CTL.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.num_gpus = 8
        FLAGS.train_epochs = 17
        FLAGS.batch_size = 1048576
        FLAGS.eval_batch_size = 1048000
        FLAGS.learning_rate = 0.0045
        FLAGS.beta1 = 0.25
        FLAGS.beta2 = 0.5
        FLAGS.epsilon = 1e-08
        FLAGS.train_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'training_cycle_*/*')
        FLAGS.eval_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'eval_data/*')
        FLAGS.input_meta_data_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'meta_data.json')
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_8_gpu_tf_data_fp16_mlperf_like(self):
        if False:
            i = 10
            return i + 15
        '8 GPU FP16'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.train_epochs = 17
        FLAGS.batch_size = 1048576
        FLAGS.eval_batch_size = 1048000
        FLAGS.learning_rate = 0.0045
        FLAGS.beta1 = 0.25
        FLAGS.beta2 = 0.5
        FLAGS.epsilon = 1e-08
        FLAGS.dtype = 'fp16'
        FLAGS.loss_scale = 8192
        FLAGS.train_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'training_cycle_*/*')
        FLAGS.eval_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'eval_data/*')
        FLAGS.input_meta_data_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'meta_data.json')
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_8_gpu_tf_data_ctl_fp16_mlperf_like(self):
        if False:
            while True:
                i = 10
        '8 GPU FP16 using CTL'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.num_gpus = 8
        FLAGS.train_epochs = 17
        FLAGS.batch_size = 1048576
        FLAGS.eval_batch_size = 1048000
        FLAGS.learning_rate = 0.0045
        FLAGS.beta1 = 0.25
        FLAGS.beta2 = 0.5
        FLAGS.epsilon = 1e-08
        FLAGS.dtype = 'fp16'
        FLAGS.loss_scale = 8192
        FLAGS.train_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'training_cycle_*/*')
        FLAGS.eval_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'eval_data/*')
        FLAGS.input_meta_data_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'meta_data.json')
        self._run_and_report_benchmark_mlperf_like()

    def benchmark_8_gpu_tf_data_ctl_fp16_graph_rewrite_mlperf_like(self):
        if False:
            return 10
        '8 GPU FP16 graph rewrite using CTL.'
        self._setup()
        FLAGS.keras_use_ctl = True
        FLAGS.num_gpus = 8
        FLAGS.train_epochs = 17
        FLAGS.batch_size = 1048576
        FLAGS.eval_batch_size = 1048000
        FLAGS.learning_rate = 0.0045
        FLAGS.beta1 = 0.25
        FLAGS.beta2 = 0.5
        FLAGS.epsilon = 1e-08
        FLAGS.dtype = 'fp16'
        FLAGS.fp16_implementation = 'graph_rewrite'
        FLAGS.loss_scale = 8192
        FLAGS.train_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'training_cycle_*/*')
        FLAGS.eval_dataset_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'eval_data/*')
        FLAGS.input_meta_data_path = os.path.join(NCF_TF_DATA_1M_BATCH_DIR_NAME, 'meta_data.json')
        self._run_and_report_benchmark_mlperf_like()

class NCFKerasSynth(NCFKerasBenchmarkBase):
    """Benchmark NCF model using synthetic data."""

    def __init__(self, output_dir=None, default_flags=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        default_flags = {}
        default_flags['dataset'] = 'ml-20m'
        default_flags['num_gpus'] = 1
        default_flags['train_epochs'] = 8
        default_flags['batch_size'] = 99000
        default_flags['eval_batch_size'] = 160000
        default_flags['learning_rate'] = 0.00382059
        default_flags['beta1'] = 0.783529
        default_flags['beta2'] = 0.909003
        default_flags['epsilon'] = 1.45439e-07
        default_flags['layers'] = [256, 256, 128, 64]
        default_flags['num_factors'] = 64
        default_flags['hr_threshold'] = 0.635
        default_flags['use_synthetic_data'] = True
        super(NCFKerasSynth, self).__init__(output_dir=output_dir, default_flags=default_flags, **kwargs)

    def benchmark_1_gpu(self):
        if False:
            print('Hello World!')
        self._setup()
        self._run_and_report_benchmark()

    def benchmark_2_gpus(self):
        if False:
            while True:
                i = 10
        self._setup()
        FLAGS.num_gpus = 2
        self._run_and_report_benchmark()
if __name__ == '__main__':
    tf.test.main()