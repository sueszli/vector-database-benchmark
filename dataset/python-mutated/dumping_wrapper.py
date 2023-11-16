"""Debugger wrapper session that dumps debug data to file:// URLs."""
import os
import threading
import time
from tensorflow.core.util import event_pb2
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.wrappers import framework
from tensorflow.python.platform import gfile

class DumpingDebugWrapperSession(framework.NonInteractiveDebugWrapperSession):
    """Debug Session wrapper that dumps debug data to filesystem."""

    def __init__(self, sess, session_root, watch_fn=None, thread_name_filter=None, pass_through_operrors=None):
        if False:
            print('Hello World!')
        "Constructor of DumpingDebugWrapperSession.\n\n    Args:\n      sess: The TensorFlow `Session` object being wrapped.\n      session_root: (`str`) Path to the session root directory. Must be a\n        directory that does not exist or an empty directory. If the directory\n        does not exist, it will be created by the debugger core during debug\n        `tf.Session.run`\n        calls.\n        As the `run()` calls occur, subdirectories will be added to\n        `session_root`. The subdirectories' names has the following pattern:\n          run_<epoch_time_stamp>_<zero_based_run_counter>\n        E.g., run_1480734393835964_ad4c953a85444900ae79fc1b652fb324\n      watch_fn: (`Callable`) A Callable that can be used to define per-run\n        debug ops and watched tensors. See the doc of\n        `NonInteractiveDebugWrapperSession.__init__()` for details.\n      thread_name_filter: Regular-expression white list for threads on which the\n        wrapper session will be active. See doc of `BaseDebugWrapperSession` for\n        more details.\n      pass_through_operrors: If true, all captured OpErrors will be\n        propagated. By default this captures all OpErrors.\n\n    Raises:\n       ValueError: If `session_root` is an existing and non-empty directory or\n       if `session_root` is a file.\n    "
        framework.NonInteractiveDebugWrapperSession.__init__(self, sess, watch_fn=watch_fn, thread_name_filter=thread_name_filter, pass_through_operrors=pass_through_operrors)
        session_root = os.path.expanduser(session_root)
        if gfile.Exists(session_root):
            if not gfile.IsDirectory(session_root):
                raise ValueError('session_root path points to a file: %s' % session_root)
            elif gfile.ListDirectory(session_root):
                raise ValueError('session_root path points to a non-empty directory: %s' % session_root)
        else:
            gfile.MakeDirs(session_root)
        self._session_root = session_root
        self._run_counter = 0
        self._run_counter_lock = threading.Lock()

    def prepare_run_debug_urls(self, fetches, feed_dict):
        if False:
            print('Hello World!')
        'Implementation of abstract method in superclass.\n\n    See doc of `NonInteractiveDebugWrapperSession.prepare_run_debug_urls()`\n    for details. This implementation creates a run-specific subdirectory under\n    self._session_root and stores information regarding run `fetches` and\n    `feed_dict.keys()` in the subdirectory.\n\n    Args:\n      fetches: Same as the `fetches` argument to `Session.run()`\n      feed_dict: Same as the `feed_dict` argument to `Session.run()`\n\n    Returns:\n      debug_urls: (`str` or `list` of `str`) file:// debug URLs to be used in\n        this `Session.run()` call.\n    '
        self._run_counter_lock.acquire()
        run_dir = os.path.join(self._session_root, 'run_%d_%d' % (int(time.time() * 1000000.0), self._run_counter))
        self._run_counter += 1
        self._run_counter_lock.release()
        gfile.MkDir(run_dir)
        fetches_event = event_pb2.Event()
        fetches_event.log_message.message = repr(fetches)
        fetches_path = os.path.join(run_dir, debug_data.METADATA_FILE_PREFIX + debug_data.FETCHES_INFO_FILE_TAG)
        with gfile.Open(os.path.join(fetches_path), 'wb') as f:
            f.write(fetches_event.SerializeToString())
        feed_keys_event = event_pb2.Event()
        feed_keys_event.log_message.message = repr(feed_dict.keys()) if feed_dict else repr(feed_dict)
        feed_keys_path = os.path.join(run_dir, debug_data.METADATA_FILE_PREFIX + debug_data.FEED_KEYS_INFO_FILE_TAG)
        with gfile.Open(os.path.join(feed_keys_path), 'wb') as f:
            f.write(feed_keys_event.SerializeToString())
        return ['file://' + run_dir]