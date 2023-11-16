"""TensorFlow 2.x Profiler.

The profiler has two modes:
- Programmatic Mode: start(logdir), stop(), and Profiler class. Profiling starts
                     when calling start(logdir) or create a Profiler class.
                     Profiling stops when calling stop() to save to
                     TensorBoard logdir or destroying the Profiler class.
- Sampling Mode: start_server(). It will perform profiling after receiving a
                 profiling request.

NOTE: Only one active profiler session is allowed. Use of simultaneous
Programmatic Mode and Sampling Mode is undefined and will likely fail.

NOTE: The Keras TensorBoard callback will automatically perform sampled
profiling. Before enabling customized profiling, set the callback flag
"profile_batches=[]" to disable automatic sampled profiling.
"""
import collections
import threading
from tensorflow.python.framework import errors
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.tf_export import tf_export
_profiler = None
_profiler_lock = threading.Lock()

@tf_export('profiler.experimental.ProfilerOptions', v1=[])
class ProfilerOptions(collections.namedtuple('ProfilerOptions', ['host_tracer_level', 'python_tracer_level', 'device_tracer_level', 'delay_ms'])):
    """Options for finer control over the profiler.

  Use `tf.profiler.experimental.ProfilerOptions` to control `tf.profiler`
  behavior.

  Fields:
    host_tracer_level: Adjust CPU tracing level. Values are: `1` - critical info
      only, `2` - info, `3` - verbose. [default value is `2`]
    python_tracer_level: Toggle tracing of Python function calls. Values are:
      `1` - enabled, `0` - disabled [default value is `0`]
    device_tracer_level: Adjust device (TPU/GPU) tracing level. Values are:
      `1` - enabled, `0` - disabled [default value is `1`]
    delay_ms: Requests for all hosts to start profiling at a timestamp that is
      `delay_ms` away from the current time. `delay_ms` is in milliseconds. If
      zero, each host will start profiling immediately upon receiving the
      request. Default value is `None`, allowing the profiler guess the best
      value.
  """

    def __new__(cls, host_tracer_level=2, python_tracer_level=0, device_tracer_level=1, delay_ms=None):
        if False:
            i = 10
            return i + 15
        return super(ProfilerOptions, cls).__new__(cls, host_tracer_level, python_tracer_level, device_tracer_level, delay_ms)

@tf_export('profiler.experimental.start', v1=[])
def start(logdir, options=None):
    if False:
        for i in range(10):
            print('nop')
    "Start profiling TensorFlow performance.\n\n  Args:\n    logdir: Profiling results log directory.\n    options: `ProfilerOptions` namedtuple to specify miscellaneous profiler\n      options. See example usage below.\n\n  Raises:\n    AlreadyExistsError: If a profiling session is already running.\n\n  Example usage:\n  ```python\n  options = tf.profiler.experimental.ProfilerOptions(host_tracer_level = 3,\n                                                     python_tracer_level = 1,\n                                                     device_tracer_level = 1)\n  tf.profiler.experimental.start('logdir_path', options = options)\n  # Training code here\n  tf.profiler.experimental.stop()\n  ```\n\n  To view the profiling results, launch TensorBoard and point it to `logdir`.\n  Open your browser and go to `localhost:6006/#profile` to view profiling\n  results.\n\n  "
    global _profiler
    with _profiler_lock:
        if _profiler is not None:
            raise errors.AlreadyExistsError(None, None, 'Another profiler is running.')
        _profiler = _pywrap_profiler.ProfilerSession()
        try:
            opts = dict(options._asdict()) if options is not None else {}
            _profiler.start(logdir, opts)
        except errors.AlreadyExistsError:
            logging.warning('Another profiler session is running which is probably created by profiler server. Please avoid using profiler server and profiler APIs at the same time.')
            raise errors.AlreadyExistsError(None, None, 'Another profiler is running.')
        except Exception:
            _profiler = None
            raise

@tf_export('profiler.experimental.stop', v1=[])
def stop(save=True):
    if False:
        return 10
    'Stops the current profiling session.\n\n  The profiler session will be stopped and profile results can be saved.\n\n  Args:\n    save: An optional variable to save the results to TensorBoard. Default True.\n\n  Raises:\n    UnavailableError: If there is no active profiling session.\n  '
    global _profiler
    with _profiler_lock:
        if _profiler is None:
            raise errors.UnavailableError(None, None, 'Cannot export profiling results. No profiler is running.')
        if save:
            try:
                _profiler.export_to_tb()
            except Exception:
                _profiler = None
                raise
        _profiler = None

def warmup():
    if False:
        while True:
            i = 10
    'Warm-up the profiler session.\n\n  The profiler session will set up profiling context, including loading CUPTI\n  library for GPU profiling. This is used for improving the accuracy of\n  the profiling results.\n\n  '
    start('')
    stop(save=False)

@tf_export('profiler.experimental.server.start', v1=[])
def start_server(port):
    if False:
        while True:
            i = 10
    'Start a profiler grpc server that listens to given port.\n\n  The profiler server will exit when the process finishes. The service is\n  defined in tensorflow/core/profiler/profiler_service.proto.\n\n  Args:\n    port: port profiler server listens to.\n  Example usage: ```python tf.profiler.experimental.server.start(6009) # do\n    your training here.\n  '
    _pywrap_profiler.start_server(port)

@tf_export('profiler.experimental.Profile', v1=[])
class Profile(object):
    """Context-manager profile API.

  Profiling will start when entering the scope, and stop and save the results to
  the logdir when exits the scope. Open TensorBoard profile tab to view results.

  Example usage:
  ```python
  with tf.profiler.experimental.Profile("/path/to/logdir"):
    # do some work
  ```
  """

    def __init__(self, logdir, options=None):
        if False:
            while True:
                i = 10
        "Creates a context manager object for profiler API.\n\n    Args:\n      logdir: profile data will save to this directory.\n      options: An optional `tf.profiler.experimental.ProfilerOptions` can be\n        provided to fine tune the profiler's behavior.\n    "
        self._logdir = logdir
        self._options = options

    def __enter__(self):
        if False:
            for i in range(10):
                print('nop')
        start(self._logdir, self._options)

    def __exit__(self, typ, value, tb):
        if False:
            i = 10
            return i + 15
        stop()