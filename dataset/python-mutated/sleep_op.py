"""A wrapper for gen_sleep_op.py.

This defines a public API (and provides a docstring for it) for the C++ Op
defined by sleep_kernel.cc
"""
import tensorflow as tf
from tensorflow.python.platform import resource_loader
_sleep_module = tf.load_op_library(resource_loader.get_path_to_datafile('sleep_kernel.so'))
examples_async_sleep = _sleep_module.examples_async_sleep
examples_sync_sleep = _sleep_module.examples_sync_sleep

def AsyncSleep(delay, name=None):
    if False:
        return 10
    'Pause for `delay` seconds (which need not be an integer).\n\n  This is an asynchronous (non-blocking) version of a sleep op. It includes\n  any time spent being blocked by another thread in `delay`. If it is blocked\n  for a fraction of the time specified by `delay`, it only calls `sleep`\n  (actually `usleep`) only for the remainder. If it is blocked for the full\n  time specified by `delay` or more, it returns without explictly calling\n  `sleep`.\n\n  Args:\n    delay: tf.Tensor which is a scalar of type float.\n    name: An optional name for the op.\n\n  Returns:\n    The `delay` value.\n  '
    return examples_async_sleep(delay=delay, name=name)

def SyncSleep(delay, name=None):
    if False:
        print('Hello World!')
    "Pause for `delay` seconds (which need not be an integer).\n\n  This is a synchronous (blocking) version of a sleep op. It's purpose is\n  to be contrasted with Examples>AsyncSleep.\n\n  Args:\n    delay: tf.Tensor which is a scalar of type float.\n    name: An optional name for the op.\n\n  Returns:\n    The `delay` value.\n  "
    return examples_sync_sleep(delay=delay, name=name)