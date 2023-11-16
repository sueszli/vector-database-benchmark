"""Executes Transformer w/Estimator benchmark and accuracy tests."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import time
from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
from official.transformer import transformer_main as transformer_main
from official.utils.flags import core as flags_core
from official.utils.logs import hooks
TRANSFORMER_EN2DE_DATA_DIR_NAME = 'wmt32k-en2de-official'
EN2DE_2014_BLEU_DATA_DIR_NAME = 'newstest2014'
FLAGS = flags.FLAGS

class EstimatorBenchmark(tf.test.Benchmark):
    """Methods common to executing transformer w/Estimator tests.

     Code under test for the Transformer Estimator models report the same data
     and require the same FLAG setup.
  """
    local_flags = None

    def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
        if False:
            return 10
        if not output_dir:
            output_dir = '/tmp'
        self.output_dir = output_dir
        self.default_flags = default_flags or {}
        self.flag_methods = flag_methods or {}

    def _get_model_dir(self, folder_name):
        if False:
            for i in range(10):
                print('nop')
        'Returns directory to store info, e.g. saved model and event log.'
        return os.path.join(self.output_dir, folder_name)

    def _setup(self):
        if False:
            while True:
                i = 10
        'Sets up and resets flags before each test.'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        if EstimatorBenchmark.local_flags is None:
            for flag_method in self.flag_methods:
                flag_method()
            flags.FLAGS(['foo'])
            for (k, v) in self.default_flags.items():
                setattr(FLAGS, k, v)
            saved_flag_values = flagsaver.save_flag_values()
            EstimatorBenchmark.local_flags = saved_flag_values
        else:
            flagsaver.restore_flag_values(EstimatorBenchmark.local_flags)

    def _report_benchmark(self, stats, wall_time_sec, bleu_max=None, bleu_min=None):
        if False:
            return 10
        'Report benchmark results by writing to local protobuf file.\n\n    Args:\n      stats: dict returned from estimator models with known entries.\n      wall_time_sec: the during of the benchmark execution in seconds.\n      bleu_max: highest passing level for bleu score.\n      bleu_min: lowest passing level for bleu score.\n    '
        examples_per_sec_hook = None
        for hook in stats['train_hooks']:
            if isinstance(hook, hooks.ExamplesPerSecondHook):
                examples_per_sec_hook = hook
                break
        eval_results = stats['eval_results']
        metrics = []
        if 'bleu_uncased' in stats:
            metrics.append({'name': 'bleu_uncased', 'value': stats['bleu_uncased'], 'min_value': bleu_min, 'max_value': bleu_max})
        if examples_per_sec_hook:
            exp_per_second_list = examples_per_sec_hook.current_examples_per_sec_list
            exp_per_sec = sum(exp_per_second_list) / len(exp_per_second_list)
            metrics.append({'name': 'exp_per_second', 'value': exp_per_sec})
        flags_str = flags_core.get_nondefault_flags_as_str()
        self.report_benchmark(iters=eval_results['global_step'], wall_time=wall_time_sec, metrics=metrics, extras={'flags': flags_str})

class TransformerBigEstimatorAccuracy(EstimatorBenchmark):
    """Benchmark accuracy tests for Transformer Big model w/Estimator."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            for i in range(10):
                print('nop')
        'Benchmark accuracy tests for Transformer Big model w/Estimator.\n\n    Args:\n      output_dir: directory where to output, e.g. log files.\n      root_data_dir: directory under which to look for dataset.\n      **kwargs: arbitrary named arguments. This is needed to make the\n                constructor forward compatible in case PerfZero provides more\n                named arguments before updating the constructor.\n    '
        flag_methods = [transformer_main.define_transformer_flags]
        self.train_data_dir = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME)
        self.vocab_file = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME, 'vocab.ende.32768')
        self.bleu_source = os.path.join(root_data_dir, EN2DE_2014_BLEU_DATA_DIR_NAME, 'newstest2014.en')
        self.bleu_ref = os.path.join(root_data_dir, EN2DE_2014_BLEU_DATA_DIR_NAME, 'newstest2014.de')
        super(TransformerBigEstimatorAccuracy, self).__init__(output_dir=output_dir, flag_methods=flag_methods)

    def benchmark_graph_8_gpu(self):
        if False:
            print('Hello World!')
        'Benchmark graph mode 8 gpus.\n\n      SOTA is 28.4 BLEU (uncased).\n    '
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.data_dir = self.train_data_dir
        FLAGS.vocab_file = self.vocab_file
        FLAGS['bleu_source'].value = self.bleu_source
        FLAGS['bleu_ref'].value = self.bleu_ref
        FLAGS.param_set = 'big'
        FLAGS.batch_size = 3072 * 8
        FLAGS.train_steps = 100000
        FLAGS.steps_between_evals = 5000
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
        FLAGS.hooks = ['ExamplesPerSecondHook']
        self._run_and_report_benchmark()

    def benchmark_graph_8_gpu_static_batch(self):
        if False:
            return 10
        'Benchmark graph mode 8 gpus.\n\n      SOTA is 28.4 BLEU (uncased).\n    '
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.data_dir = self.train_data_dir
        FLAGS.vocab_file = self.vocab_file
        FLAGS['bleu_source'].value = self.bleu_source
        FLAGS['bleu_ref'].value = self.bleu_ref
        FLAGS.param_set = 'big'
        FLAGS.batch_size = 3072 * 8
        FLAGS.static_batch = True
        FLAGS.max_length = 64
        FLAGS.train_steps = 100000
        FLAGS.steps_between_evals = 5000
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
        FLAGS.hooks = ['ExamplesPerSecondHook']
        self._run_and_report_benchmark()

    def _run_and_report_benchmark(self, bleu_min=28.3, bleu_max=29):
        if False:
            print('Hello World!')
        'Run benchmark and report results.\n\n    Args:\n      bleu_min: minimum expected uncased bleu. default is SOTA.\n      bleu_max: max expected uncased bleu. default is a high number.\n    '
        start_time_sec = time.time()
        stats = transformer_main.run_transformer(flags.FLAGS)
        wall_time_sec = time.time() - start_time_sec
        self._report_benchmark(stats, wall_time_sec, bleu_min=bleu_min, bleu_max=bleu_max)

class TransformerBaseEstimatorAccuracy(EstimatorBenchmark):
    """Benchmark accuracy tests for Transformer Base model w/ Estimator."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            i = 10
            return i + 15
        'Benchmark accuracy tests for Transformer Base model w/ Estimator.\n\n    Args:\n      output_dir: directory where to output e.g. log files\n      root_data_dir: directory under which to look for dataset\n      **kwargs: arbitrary named arguments. This is needed to make the\n                constructor forward compatible in case PerfZero provides more\n                named arguments before updating the constructor.\n    '
        flag_methods = [transformer_main.define_transformer_flags]
        self.train_data_dir = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME)
        self.vocab_file = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME, 'vocab.ende.32768')
        self.bleu_source = os.path.join(root_data_dir, EN2DE_2014_BLEU_DATA_DIR_NAME, 'newstest2014.en')
        self.bleu_ref = os.path.join(root_data_dir, EN2DE_2014_BLEU_DATA_DIR_NAME, 'newstest2014.de')
        super(TransformerBaseEstimatorAccuracy, self).__init__(output_dir=output_dir, flag_methods=flag_methods)

    def benchmark_graph_2_gpu(self):
        if False:
            return 10
        'Benchmark graph mode 2 gpus.\n\n      The paper uses 8 GPUs and a much larger effective batch size, this is will\n      not converge to the 27.3 BLEU (uncased) SOTA.\n    '
        self._setup()
        FLAGS.num_gpus = 2
        FLAGS.data_dir = self.train_data_dir
        FLAGS.vocab_file = self.vocab_file
        FLAGS['bleu_source'].value = self.bleu_source
        FLAGS['bleu_ref'].value = self.bleu_ref
        FLAGS.param_set = 'base'
        FLAGS.batch_size = 4096 * 2
        FLAGS.train_steps = 100000
        FLAGS.steps_between_evals = 5000
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_2_gpu')
        FLAGS.hooks = ['ExamplesPerSecondHook']
        self._run_and_report_benchmark(bleu_min=25.3, bleu_max=26)

    def benchmark_graph_8_gpu(self):
        if False:
            for i in range(10):
                print('nop')
        'Benchmark graph mode 8 gpus.\n\n      SOTA is 27.3 BLEU (uncased).\n      Best so far is 27.2  with 4048*8 at 75,000 steps.\n      27.009 with 4096*8 at 100,000 steps and earlier.\n      Other test: 2024 * 8 peaked at 26.66 at 100,000 steps.\n    '
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.data_dir = self.train_data_dir
        FLAGS.vocab_file = self.vocab_file
        FLAGS['bleu_source'].value = self.bleu_source
        FLAGS['bleu_ref'].value = self.bleu_ref
        FLAGS.param_set = 'base'
        FLAGS.batch_size = 4096 * 8
        FLAGS.train_steps = 100000
        FLAGS.steps_between_evals = 5000
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
        FLAGS.hooks = ['ExamplesPerSecondHook']
        self._run_and_report_benchmark()

    def benchmark_graph_8_gpu_static_batch(self):
        if False:
            while True:
                i = 10
        'Benchmark graph mode 8 gpus.\n\n      SOTA is 27.3 BLEU (uncased).\n      Best so far is 27.2  with 4048*8 at 75,000 steps.\n      27.009 with 4096*8 at 100,000 steps and earlier.\n      Other test: 2024 * 8 peaked at 26.66 at 100,000 steps.\n    '
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.data_dir = self.train_data_dir
        FLAGS.vocab_file = self.vocab_file
        FLAGS['bleu_source'].value = self.bleu_source
        FLAGS['bleu_ref'].value = self.bleu_ref
        FLAGS.param_set = 'base'
        FLAGS.batch_size = 4096 * 8
        FLAGS.static_batch = True
        FLAGS.max_length = 64
        FLAGS.train_steps = 100000
        FLAGS.steps_between_evals = 5000
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
        FLAGS.hooks = ['ExamplesPerSecondHook']
        self._run_and_report_benchmark()

    def benchmark_graph_fp16_8_gpu(self):
        if False:
            i = 10
            return i + 15
        'benchmark 8 gpus with fp16 mixed precision.\n\n      SOTA is 27.3 BLEU (uncased).\n    '
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.dtype = 'fp16'
        FLAGS.data_dir = self.train_data_dir
        FLAGS.vocab_file = self.vocab_file
        FLAGS['bleu_source'].value = self.bleu_source
        FLAGS['bleu_ref'].value = self.bleu_ref
        FLAGS.param_set = 'base'
        FLAGS.batch_size = 4096 * 8
        FLAGS.train_steps = 100000
        FLAGS.steps_between_evals = 5000
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_fp16_8_gpu')
        FLAGS.hooks = ['ExamplesPerSecondHook']
        self._run_and_report_benchmark()

    def _run_and_report_benchmark(self, bleu_min=27.3, bleu_max=28):
        if False:
            return 10
        'Run benchmark and report results.\n\n    Args:\n      bleu_min: minimum expected uncased bleu. default is SOTA.\n      bleu_max: max expected uncased bleu. default is a high number.\n    '
        start_time_sec = time.time()
        stats = transformer_main.run_transformer(flags.FLAGS)
        wall_time_sec = time.time() - start_time_sec
        self._report_benchmark(stats, wall_time_sec, bleu_min=bleu_min, bleu_max=bleu_max)

class TransformerEstimatorBenchmark(EstimatorBenchmark):
    """Benchmarks for Transformer (Base and Big) using Estimator."""

    def __init__(self, output_dir=None, default_flags=None, batch_per_gpu=4096):
        if False:
            print('Hello World!')
        'Initialize.\n\n    Args:\n      output_dir: Based directory for saving artifacts, e.g. checkpoints.\n      default_flags: default flags to use for all tests.\n      batch_per_gpu: batch size to use per gpu.\n    '
        flag_methods = [transformer_main.define_transformer_flags]
        self.batch_per_gpu = batch_per_gpu
        super(TransformerEstimatorBenchmark, self).__init__(output_dir=output_dir, default_flags=default_flags, flag_methods=flag_methods)

    def benchmark_graph_1_gpu(self):
        if False:
            while True:
                i = 10
        'Benchmark graph 1 gpu.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.batch_size = self.batch_per_gpu
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_1_gpu')
        self._run_and_report_benchmark()

    def benchmark_graph_fp16_1_gpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark graph fp16 1 gpu.'
        self._setup()
        FLAGS.num_gpus = 1
        FLAGS.dtype = 'fp16'
        FLAGS.batch_size = self.batch_per_gpu
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_fp16_1_gpu')
        self._run_and_report_benchmark()

    def benchmark_graph_2_gpu(self):
        if False:
            while True:
                i = 10
        'Benchmark graph 2 gpus.'
        self._setup()
        FLAGS.num_gpus = 2
        FLAGS.batch_size = self.batch_per_gpu * 2
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_2_gpu')
        self._run_and_report_benchmark()

    def benchmark_graph_fp16_2_gpu(self):
        if False:
            print('Hello World!')
        'Benchmark graph fp16 2 gpus.'
        self._setup()
        FLAGS.num_gpus = 2
        FLAGS.dtype = 'fp16'
        FLAGS.batch_size = self.batch_per_gpu * 2
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_fp16_2_gpu')
        self._run_and_report_benchmark()

    def benchmark_graph_4_gpu(self):
        if False:
            while True:
                i = 10
        'Benchmark graph 4 gpus.'
        self._setup()
        FLAGS.num_gpus = 4
        FLAGS.batch_size = self.batch_per_gpu * 4
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_4_gpu')
        self._run_and_report_benchmark()

    def benchmark_graph_fp16_4_gpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark 4 graph fp16 gpus.'
        self._setup()
        FLAGS.num_gpus = 4
        FLAGS.dtype = 'fp16'
        FLAGS.batch_size = self.batch_per_gpu * 4
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_fp16_4_gpu')
        self._run_and_report_benchmark()

    def benchmark_graph_8_gpu(self):
        if False:
            i = 10
            return i + 15
        'Benchmark graph 8 gpus.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.batch_size = self.batch_per_gpu * 8
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_8_gpu')
        self._run_and_report_benchmark()

    def benchmark_graph_fp16_8_gpu(self):
        if False:
            return 10
        'Benchmark graph fp16 8 gpus.'
        self._setup()
        FLAGS.num_gpus = 8
        FLAGS.dtype = 'fp16'
        FLAGS.batch_size = self.batch_per_gpu * 8
        FLAGS.model_dir = self._get_model_dir('benchmark_graph_fp16_8_gpu')
        self._run_and_report_benchmark()

    def _run_and_report_benchmark(self):
        if False:
            return 10
        start_time_sec = time.time()
        stats = transformer_main.run_transformer(flags.FLAGS)
        wall_time_sec = time.time() - start_time_sec
        self._report_benchmark(stats, wall_time_sec)

class TransformerBaseEstimatorBenchmarkSynth(TransformerEstimatorBenchmark):
    """Transformer based version synthetic benchmark tests."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            print('Hello World!')
        def_flags = {}
        def_flags['param_set'] = 'base'
        def_flags['use_synthetic_data'] = True
        def_flags['train_steps'] = 200
        def_flags['steps_between_evals'] = 200
        def_flags['hooks'] = ['ExamplesPerSecondHook']
        super(TransformerBaseEstimatorBenchmarkSynth, self).__init__(output_dir=output_dir, default_flags=def_flags)

class TransformerBaseEstimatorBenchmarkReal(TransformerEstimatorBenchmark):
    """Transformer based version real data benchmark tests."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            return 10
        train_data_dir = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME)
        vocab_file = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME, 'vocab.ende.32768')
        def_flags = {}
        def_flags['param_set'] = 'base'
        def_flags['vocab_file'] = vocab_file
        def_flags['data_dir'] = train_data_dir
        def_flags['train_steps'] = 200
        def_flags['steps_between_evals'] = 200
        def_flags['hooks'] = ['ExamplesPerSecondHook']
        super(TransformerBaseEstimatorBenchmarkReal, self).__init__(output_dir=output_dir, default_flags=def_flags)

class TransformerBigEstimatorBenchmarkReal(TransformerEstimatorBenchmark):
    """Transformer based version real data benchmark tests."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            return 10
        train_data_dir = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME)
        vocab_file = os.path.join(root_data_dir, TRANSFORMER_EN2DE_DATA_DIR_NAME, 'vocab.ende.32768')
        def_flags = {}
        def_flags['param_set'] = 'big'
        def_flags['vocab_file'] = vocab_file
        def_flags['data_dir'] = train_data_dir
        def_flags['train_steps'] = 200
        def_flags['steps_between_evals'] = 200
        def_flags['hooks'] = ['ExamplesPerSecondHook']
        super(TransformerBigEstimatorBenchmarkReal, self).__init__(output_dir=output_dir, default_flags=def_flags, batch_per_gpu=3072)

class TransformerBigEstimatorBenchmarkSynth(TransformerEstimatorBenchmark):
    """Transformer based version synthetic benchmark tests."""

    def __init__(self, output_dir=None, root_data_dir=None, **kwargs):
        if False:
            i = 10
            return i + 15
        def_flags = {}
        def_flags['param_set'] = 'big'
        def_flags['use_synthetic_data'] = True
        def_flags['train_steps'] = 200
        def_flags['steps_between_evals'] = 200
        def_flags['hooks'] = ['ExamplesPerSecondHook']
        super(TransformerBigEstimatorBenchmarkSynth, self).__init__(output_dir=output_dir, default_flags=def_flags, batch_per_gpu=3072)