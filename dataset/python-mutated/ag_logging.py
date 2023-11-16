"""Logging and debugging utilities."""
import os
import sys
import traceback
import logging
from nvidia.dali._autograph.utils.all_utils import export_symbol
VERBOSITY_VAR_NAME = 'AUTOGRAPH_VERBOSITY'
DEFAULT_VERBOSITY = 0
verbosity_level = None
echo_log_to_stdout = False
if hasattr(sys, 'ps1') or hasattr(sys, 'ps2'):
    echo_log_to_stdout = True

@export_symbol('autograph.set_verbosity')
def set_verbosity(level, alsologtostdout=False):
    if False:
        for i in range(10):
            print('nop')
    "Sets the AutoGraph verbosity level.\n\n  _Debug logging in AutoGraph_\n\n  More verbose logging is useful to enable when filing bug reports or doing\n  more in-depth debugging.\n\n  There are two means to control the logging verbosity:\n\n   * The `set_verbosity` function\n\n   * The `AUTOGRAPH_VERBOSITY` environment variable\n\n  `set_verbosity` takes precedence over the environment variable.\n\n  For example:\n\n  ```python\n  import os\n  import tensorflow as tf\n\n  os.environ['AUTOGRAPH_VERBOSITY'] = '5'\n  # Verbosity is now 5\n\n  tf.autograph.set_verbosity(0)\n  # Verbosity is now 0\n\n  os.environ['AUTOGRAPH_VERBOSITY'] = '1'\n  # No effect, because set_verbosity was already called.\n  ```\n\n  Logs entries are output to [absl](https://abseil.io)'s\n  [default output](https://abseil.io/docs/python/guides/logging),\n  with `INFO` level.\n  Logs can be mirrored to stdout by using the `alsologtostdout` argument.\n  Mirroring is enabled by default when Python runs in interactive mode.\n\n  Args:\n    level: int, the verbosity level; larger values specify increased verbosity;\n      0 means no logging. When reporting bugs, it is recommended to set this\n      value to a larger number, like 10.\n    alsologtostdout: bool, whether to also output log messages to `sys.stdout`.\n  "
    global verbosity_level
    global echo_log_to_stdout
    verbosity_level = level
    echo_log_to_stdout = alsologtostdout

@export_symbol('autograph.trace')
def trace(*args):
    if False:
        print('Hello World!')
    'Traces argument information at compilation time.\n\n  `trace` is useful when debugging, and it always executes during the tracing\n  phase, that is, when the TF graph is constructed.\n\n  _Example usage_\n\n  ```python\n  import tensorflow as tf\n\n  for i in tf.range(10):\n    tf.autograph.trace(i)\n  # Output: <Tensor ...>\n  ```\n\n  Args:\n    *args: Arguments to print to `sys.stdout`.\n  '
    print(*args)

def get_verbosity():
    if False:
        print('Hello World!')
    global verbosity_level
    if verbosity_level is not None:
        return verbosity_level
    return int(os.getenv(VERBOSITY_VAR_NAME, DEFAULT_VERBOSITY))

def has_verbosity(level):
    if False:
        return 10
    return get_verbosity() >= level

def _output_to_stdout(msg, *args, **kwargs):
    if False:
        for i in range(10):
            print('nop')
    print(msg % args)
    if kwargs.get('exc_info', False):
        traceback.print_exc()

def error(level, msg, *args, **kwargs):
    if False:
        print('Hello World!')
    if has_verbosity(level):
        logging.error(msg, *args, **kwargs)
        if echo_log_to_stdout:
            _output_to_stdout('ERROR: ' + msg, *args, **kwargs)

def log(level, msg, *args, **kwargs):
    if False:
        print('Hello World!')
    if has_verbosity(level):
        logging.info(msg, *args, **kwargs)
        if echo_log_to_stdout:
            _output_to_stdout(msg, *args, **kwargs)

def warning(msg, *args, **kwargs):
    if False:
        return 10
    logging.warning(msg, *args, **kwargs)
    if echo_log_to_stdout:
        _output_to_stdout('WARNING: ' + msg, *args, **kwargs)
        sys.stdout.flush()