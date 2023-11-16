"""Test runner for TensorFlow tests."""
import os
import shlex
import sys
import time
from absl import app
from absl import flags
from google.protobuf import json_format
from google.protobuf import text_format
from tensorflow.core.util import test_log_pb2
from tensorflow.python.platform import gfile
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
from tensorflow.tools.test import run_and_gather_logs_lib
try:
    import cpuinfo
    import psutil
except ImportError as e:
    tf_logging.error('\n\n\nERROR: Unable to import necessary library: {}.  Issuing a soft exit.\n\n\n'.format(e))
    sys.exit(0)
FLAGS = flags.FLAGS
flags.DEFINE_string('name', '', 'Benchmark target identifier.')
flags.DEFINE_string('test_name', '', 'Test target to run.')
flags.DEFINE_multi_string('test_args', '', 'Test arguments, space separated. May be specified more than once, in which case\nthe args are all appended.')
flags.DEFINE_boolean('test_log_output_use_tmpdir', False, 'Whether to store the log output into tmpdir.')
flags.DEFINE_string('benchmark_type', '', 'Benchmark type (BenchmarkType enum string).')
flags.DEFINE_string('compilation_mode', '', 'Mode used during this build (e.g. opt, dbg).')
flags.DEFINE_string('cc_flags', '', 'CC flags used during this build.')
flags.DEFINE_string('test_log_output_dir', '', 'Directory for benchmark results output.')
flags.DEFINE_string('test_log_output_filename', '', 'Filename to write output benchmark results to. If the filename\n                    is not specified, it will be automatically created.')
flags.DEFINE_boolean('skip_export', False, 'Whether to skip exporting test results.')

def gather_build_configuration():
    if False:
        i = 10
        return i + 15
    build_config = test_log_pb2.BuildConfiguration()
    build_config.mode = FLAGS.compilation_mode
    cc_flags = [flag for flag in shlex.split(FLAGS.cc_flags) if not flag.startswith('-i')]
    build_config.cc_flags.extend(cc_flags)
    return build_config

def main(unused_args):
    if False:
        return 10
    name = FLAGS.name
    test_name = FLAGS.test_name
    test_args = ' '.join(FLAGS.test_args)
    benchmark_type = FLAGS.benchmark_type
    (test_results, _) = run_and_gather_logs_lib.run_and_gather_logs(name, test_name=test_name, test_args=test_args, benchmark_type=benchmark_type, skip_processing_logs=FLAGS.skip_export)
    if FLAGS.skip_export:
        return
    test_results.build_configuration.CopyFrom(gather_build_configuration())
    test_results.run_configuration.env_vars.update(os.environ)
    if not FLAGS.test_log_output_dir:
        print(text_format.MessageToString(test_results))
        return
    if FLAGS.test_log_output_filename:
        file_name = FLAGS.test_log_output_filename
    else:
        file_name = name.strip('/').translate(str.maketrans('/:', '__')) + time.strftime('%Y%m%d%H%M%S', time.gmtime())
    if FLAGS.test_log_output_use_tmpdir:
        tmpdir = test.get_temp_dir()
        output_path = os.path.join(tmpdir, FLAGS.test_log_output_dir, file_name)
    else:
        output_path = os.path.join(os.path.abspath(FLAGS.test_log_output_dir), file_name)
    json_test_results = json_format.MessageToJson(test_results)
    gfile.GFile(output_path + '.json', 'w').write(json_test_results)
    tf_logging.info('Test results written to: %s' % output_path)
if __name__ == '__main__':
    app.run(main)