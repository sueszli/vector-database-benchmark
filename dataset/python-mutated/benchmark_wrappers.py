"""Utils to annotate and trace benchmarks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from absl import flags
from absl import logging
from absl.testing import flagsaver
FLAGS = flags.FLAGS
flags.DEFINE_multi_string('benchmark_method_flags', None, "Optional list of runtime flags of the form key=value. Specify multiple times to specify different flags. These will override the FLAGS object directly after hardcoded settings in individual benchmark methods before they call _run_and_report benchmark. Example if we set --benchmark_method_flags=train_steps=10 and a benchmark method hardcodes FLAGS.train_steps=10000 and later calls _run_and_report_benchmark, it'll only run for 10 steps. This is useful for debugging/profiling workflows.")

def enable_runtime_flags(decorated_func):
    if False:
        return 10
    'Sets attributes from --benchmark_method_flags for method execution.\n\n  @enable_runtime_flags decorator temporarily adds flags passed in via\n  --benchmark_method_flags and runs the decorated function in that context.\n\n  A user can set --benchmark_method_flags=train_steps=5 to run the benchmark\n  method in the snippet below with FLAGS.train_steps=5 for debugging (without\n  modifying the benchmark code).\n\n  class ModelBenchmark():\n\n    @benchmark_wrappers.enable_runtime_flags\n    def _run_and_report_benchmark(self):\n      # run benchmark ...\n      # report benchmark results ...\n\n    def benchmark_method(self):\n      FLAGS.train_steps = 1000\n      ...\n      self._run_and_report_benchmark()\n\n  Args:\n    decorated_func: The method that runs the benchmark after previous setup\n      execution that set some flags.\n\n  Returns:\n    new_func: The same method which executes in a temporary context where flag\n      overrides from --benchmark_method_flags are active.\n  '

    def runner(*args, **kwargs):
        if False:
            i = 10
            return i + 15
        'Creates a temporary context to activate --benchmark_method_flags.'
        if FLAGS.benchmark_method_flags:
            saved_flag_values = flagsaver.save_flag_values()
            for key_value in FLAGS.benchmark_method_flags:
                (key, value) = key_value.split('=', 1)
                try:
                    numeric_float = float(value)
                    numeric_int = int(numeric_float)
                    if abs(numeric_int) == abs(numeric_float):
                        flag_value = numeric_int
                    else:
                        flag_value = numeric_float
                except ValueError:
                    flag_value = value
                logging.info('Setting --%s=%s', key, flag_value)
                setattr(FLAGS, key, flag_value)
        else:
            saved_flag_values = None
        try:
            result = decorated_func(*args, **kwargs)
            return result
        finally:
            if saved_flag_values:
                flagsaver.restore_flag_values(saved_flag_values)
    return runner