"""Utilities related to TensorFlow exception stack trace prettifying."""
import os
import sys
import threading
import traceback
import types
from tensorflow.python.util import tf_decorator
from tensorflow.python.util.tf_export import tf_export
_ENABLE_TRACEBACK_FILTERING = threading.local()
_EXCLUDED_PATHS = (os.path.abspath(os.path.join(__file__, '..', '..')),)

@tf_export('debugging.is_traceback_filtering_enabled')
def is_traceback_filtering_enabled():
    if False:
        print('Hello World!')
    'Check whether traceback filtering is currently enabled.\n\n  See also `tf.debugging.enable_traceback_filtering()` and\n  `tf.debugging.disable_traceback_filtering()`. Note that filtering out\n  internal frames from the tracebacks of exceptions raised by TensorFlow code\n  is the default behavior.\n\n  Returns:\n    True if traceback filtering is enabled\n    (e.g. if `tf.debugging.enable_traceback_filtering()` was called),\n    and False otherwise (e.g. if `tf.debugging.disable_traceback_filtering()`\n    was called).\n  '
    value = getattr(_ENABLE_TRACEBACK_FILTERING, 'value', True)
    return value

@tf_export('debugging.enable_traceback_filtering')
def enable_traceback_filtering():
    if False:
        return 10
    "Enable filtering out TensorFlow-internal frames in exception stack traces.\n\n  Raw TensorFlow stack traces involve many internal frames, which can be\n  challenging to read through, while not being actionable for end users.\n  By default, TensorFlow filters internal frames in most exceptions that it\n  raises, to keep stack traces short, readable, and focused on what's\n  actionable for end users (their own code).\n\n  If you have previously disabled traceback filtering via\n  `tf.debugging.disable_traceback_filtering()`, you can re-enable it via\n  `tf.debugging.enable_traceback_filtering()`.\n\n  Raises:\n    RuntimeError: If Python version is not at least 3.7.\n  "
    if sys.version_info.major != 3 or sys.version_info.minor < 7:
        raise RuntimeError(f'Traceback filtering is only available with Python 3.7 or higher. This Python version: {sys.version}')
    global _ENABLE_TRACEBACK_FILTERING
    _ENABLE_TRACEBACK_FILTERING.value = True

@tf_export('debugging.disable_traceback_filtering')
def disable_traceback_filtering():
    if False:
        for i in range(10):
            print('nop')
    "Disable filtering out TensorFlow-internal frames in exception stack traces.\n\n  Raw TensorFlow stack traces involve many internal frames, which can be\n  challenging to read through, while not being actionable for end users.\n  By default, TensorFlow filters internal frames in most exceptions that it\n  raises, to keep stack traces short, readable, and focused on what's\n  actionable for end users (their own code).\n\n  Calling `tf.debugging.disable_traceback_filtering` disables this filtering\n  mechanism, meaning that TensorFlow exceptions stack traces will include\n  all frames, in particular TensorFlow-internal ones.\n\n  **If you are debugging a TensorFlow-internal issue, you need to call\n  `tf.debugging.disable_traceback_filtering`**.\n  To re-enable traceback filtering afterwards, you can call\n  `tf.debugging.enable_traceback_filtering()`.\n  "
    global _ENABLE_TRACEBACK_FILTERING
    _ENABLE_TRACEBACK_FILTERING.value = False

def include_frame(fname):
    if False:
        for i in range(10):
            print('nop')
    for exclusion in _EXCLUDED_PATHS:
        if exclusion in fname:
            return False
    return True

def _process_traceback_frames(tb):
    if False:
        while True:
            i = 10
    new_tb = None
    tb_list = list(traceback.walk_tb(tb))
    for (f, line_no) in reversed(tb_list):
        if include_frame(f.f_code.co_filename):
            new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
    if new_tb is None and tb_list:
        (f, line_no) = tb_list[-1]
        new_tb = types.TracebackType(new_tb, f, f.f_lasti, line_no)
    return new_tb

def filter_traceback(fn):
    if False:
        print('Hello World!')
    "Decorator to filter out TF-internal stack trace frames in exceptions.\n\n  Raw TensorFlow stack traces involve many internal frames, which can be\n  challenging to read through, while not being actionable for end users.\n  By default, TensorFlow filters internal frames in most exceptions that it\n  raises, to keep stack traces short, readable, and focused on what's\n  actionable for end users (their own code).\n\n  Arguments:\n    fn: The function or method to decorate. Any exception raised within the\n      function will be reraised with its internal stack trace frames filtered\n      out.\n\n  Returns:\n    Decorated function or method.\n  "
    if sys.version_info.major != 3 or sys.version_info.minor < 7:
        return fn

    def error_handler(*args, **kwargs):
        if False:
            print('Hello World!')
        try:
            if not is_traceback_filtering_enabled():
                return fn(*args, **kwargs)
        except NameError:
            return fn(*args, **kwargs)
        filtered_tb = None
        try:
            return fn(*args, **kwargs)
        except Exception as e:
            filtered_tb = _process_traceback_frames(e.__traceback__)
            raise e.with_traceback(filtered_tb) from None
        finally:
            del filtered_tb
    return tf_decorator.make_decorator(fn, error_handler)