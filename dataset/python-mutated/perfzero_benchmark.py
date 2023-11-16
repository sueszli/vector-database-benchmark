"""Utils for creating PerfZero benchmarks."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from absl import flags
from absl.testing import flagsaver
import tensorflow as tf
FLAGS = flags.FLAGS

class PerfZeroBenchmark(tf.test.Benchmark):
    """Common methods used in PerfZero Benchmarks.

     Handles the resetting of flags between tests, loading of default_flags,
     overriding of defaults.  PerfZero (OSS) runs each test in a separate
     process reducing some need to reset the flags.
  """
    local_flags = None

    def __init__(self, output_dir=None, default_flags=None, flag_methods=None):
        if False:
            for i in range(10):
                print('nop')
        'Initialize class.\n\n    Args:\n      output_dir: Base directory to store all output for the test.\n      default_flags:\n      flag_methods:\n    '
        if not output_dir:
            output_dir = '/tmp'
        self.output_dir = output_dir
        self.default_flags = default_flags or {}
        self.flag_methods = flag_methods or {}

    def _get_model_dir(self, folder_name):
        if False:
            return 10
        'Returns directory to store info, e.g. saved model and event log.'
        return os.path.join(self.output_dir, folder_name)

    def _setup(self):
        if False:
            print('Hello World!')
        'Sets up and resets flags before each test.'
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
        if PerfZeroBenchmark.local_flags is None:
            for flag_method in self.flag_methods:
                flag_method()
            flags.FLAGS(['foo'])
            for (k, v) in self.default_flags.items():
                setattr(FLAGS, k, v)
            saved_flag_values = flagsaver.save_flag_values()
            PerfZeroBenchmark.local_flags = saved_flag_values
        else:
            flagsaver.restore_flag_values(PerfZeroBenchmark.local_flags)