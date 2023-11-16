"""Profiler client APIs."""
from tensorflow.python.framework import errors
from tensorflow.python.profiler.internal import _pywrap_profiler
from tensorflow.python.util.tf_export import tf_export
_GRPC_PREFIX = 'grpc://'

@tf_export('profiler.experimental.client.trace', v1=[])
def trace(service_addr, logdir, duration_ms, worker_list='', num_tracing_attempts=3, options=None):
    if False:
        print('Hello World!')
    "Sends gRPC requests to one or more profiler servers to perform on-demand profiling.\n\n  This method will block the calling thread until it receives responses from all\n  servers or until deadline expiration. Both single host and multiple host\n  profiling are supported on CPU, GPU, and TPU.\n  The profiled results will be saved by each server to the specified TensorBoard\n  log directory (i.e. the directory you save your model checkpoints). Use the\n  TensorBoard profile plugin to view the visualization and analysis results.\n\n  Args:\n    service_addr: A comma delimited string of gRPC addresses of the workers to\n      profile.\n      e.g. service_addr='grpc://localhost:6009'\n           service_addr='grpc://10.0.0.2:8466,grpc://10.0.0.3:8466'\n           service_addr='grpc://localhost:12345,grpc://localhost:23456'\n    logdir: Path to save profile data to, typically a TensorBoard log directory.\n      This path must be accessible to both the client and server.\n      e.g. logdir='gs://your_tb_dir'\n    duration_ms: Duration of tracing or monitoring in milliseconds. Must be\n      greater than zero.\n    worker_list: An optional TPU only configuration. The list of workers to\n      profile in the current session.\n    num_tracing_attempts: Optional. Automatically retry N times when no trace\n      event is collected (default 3).\n    options: profiler.experimental.ProfilerOptions namedtuple for miscellaneous\n      profiler options.\n\n  Raises:\n    InvalidArgumentError: For when arguments fail validation checks.\n    UnavailableError: If no trace event was collected.\n\n  Example usage (CPU/GPU):\n\n  ```python\n    # Start a profiler server before your model runs.\n    tf.profiler.experimental.server.start(6009)\n    # (Model code goes here).\n    # Send gRPC request to the profiler server to collect a trace of your model.\n    tf.profiler.experimental.client.trace('grpc://localhost:6009',\n                                          '/nfs/tb_log', 2000)\n  ```\n\n  Example usage (Multiple GPUs):\n\n  ```python\n    # E.g. your worker IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you\n    # would like to schedule start of profiling 1 second from now, for a\n    # duration of 2 seconds.\n    options['delay_ms'] = 1000\n    tf.profiler.experimental.client.trace(\n        'grpc://10.0.0.2:8466,grpc://10.0.0.3:8466,grpc://10.0.0.4:8466',\n        'gs://your_tb_dir',\n        2000,\n        options=options)\n  ```\n\n  Example usage (TPU):\n\n  ```python\n    # Send gRPC request to a TPU worker to collect a trace of your model. A\n    # profiler service has been started in the TPU worker at port 8466.\n    # E.g. your TPU IP address is 10.0.0.2 and you want to profile for 2 seconds\n    # .\n    tf.profiler.experimental.client.trace('grpc://10.0.0.2:8466',\n                                          'gs://your_tb_dir', 2000)\n  ```\n\n  Example usage (Multiple TPUs):\n\n  ```python\n    # Send gRPC request to a TPU pod to collect a trace of your model on\n    # multiple TPUs. A profiler service has been started in all the TPU workers\n    # at the port 8466.\n    # E.g. your TPU IP addresses are 10.0.0.2, 10.0.0.3, 10.0.0.4, and you want\n    # to profile for 2 seconds.\n    tf.profiler.experimental.client.trace(\n        'grpc://10.0.0.2:8466',\n        'gs://your_tb_dir',\n        2000,\n        '10.0.0.2:8466,10.0.0.3:8466,10.0.0.4:8466')\n  ```\n\n  Launch TensorBoard and point it to the same logdir you provided to this API.\n\n  ```shell\n    # logdir can be gs://your_tb_dir as in the above examples.\n    $ tensorboard --logdir=/tmp/tb_log\n  ```\n\n  Open your browser and go to localhost:6006/#profile to view profiling results.\n\n  "
    if duration_ms <= 0:
        raise errors.InvalidArgumentError(None, None, 'duration_ms must be greater than zero.')
    opts = dict(options._asdict()) if options is not None else {}
    _pywrap_profiler.trace(_strip_addresses(service_addr, _GRPC_PREFIX), logdir, worker_list, True, duration_ms, num_tracing_attempts, opts)

@tf_export('profiler.experimental.client.monitor', v1=[])
def monitor(service_addr, duration_ms, level=1):
    if False:
        print('Hello World!')
    "Sends grpc requests to profiler server to perform on-demand monitoring.\n\n  The monitoring result is a light weight performance summary of your model\n  execution. This method will block the caller thread until it receives the\n  monitoring result. This method currently supports Cloud TPU only.\n\n  Args:\n    service_addr: gRPC address of profiler service e.g. grpc://10.0.0.2:8466.\n    duration_ms: Duration of monitoring in ms.\n    level: Choose a monitoring level between 1 and 2 to monitor your job. Level\n      2 is more verbose than level 1 and shows more metrics.\n\n  Returns:\n    A string of monitoring output.\n\n  Example usage:\n\n  ```python\n    # Continuously send gRPC requests to the Cloud TPU to monitor the model\n    # execution.\n\n    for query in range(0, 100):\n      print(\n        tf.profiler.experimental.client.monitor('grpc://10.0.0.2:8466', 1000))\n  ```\n\n  "
    return _pywrap_profiler.monitor(_strip_prefix(service_addr, _GRPC_PREFIX), duration_ms, level, True)

def _strip_prefix(s, prefix):
    if False:
        i = 10
        return i + 15
    return s[len(prefix):] if s.startswith(prefix) else s

def _strip_addresses(addresses, prefix):
    if False:
        print('Hello World!')
    return ','.join([_strip_prefix(s, prefix) for s in addresses.split(',')])