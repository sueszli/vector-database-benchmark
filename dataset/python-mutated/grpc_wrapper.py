"""Debugger wrapper session that sends debug data to file:// URLs."""
import signal
import sys
import traceback
from tensorflow.python.debug.lib import common
from tensorflow.python.debug.wrappers import framework

def publish_traceback(debug_server_urls, graph, feed_dict, fetches, old_graph_version):
    if False:
        print('Hello World!')
    'Publish traceback and source code if graph version is new.\n\n  `graph.version` is compared with `old_graph_version`. If the former is higher\n  (i.e., newer), the graph traceback and the associated source code is sent to\n  the debug server at the specified gRPC URLs.\n\n  Args:\n    debug_server_urls: A single gRPC debug server URL as a `str` or a `list` of\n      debug server URLs.\n    graph: A Python `tf.Graph` object.\n    feed_dict: Feed dictionary given to the `Session.run()` call.\n    fetches: Fetches from the `Session.run()` call.\n    old_graph_version: Old graph version to compare to.\n\n  Returns:\n    If `graph.version > old_graph_version`, the new graph version as an `int`.\n    Else, the `old_graph_version` is returned.\n  '
    from tensorflow.python.debug.lib import source_remote
    if graph.version > old_graph_version:
        run_key = common.get_run_key(feed_dict, fetches)
        source_remote.send_graph_tracebacks(debug_server_urls, run_key, traceback.extract_stack(), graph, send_source=True)
        return graph.version
    else:
        return old_graph_version

class GrpcDebugWrapperSession(framework.NonInteractiveDebugWrapperSession):
    """Debug Session wrapper that send debug data to gRPC stream(s)."""

    def __init__(self, sess, grpc_debug_server_addresses, watch_fn=None, thread_name_filter=None):
        if False:
            i = 10
            return i + 15
        'Constructor of DumpingDebugWrapperSession.\n\n    Args:\n      sess: The TensorFlow `Session` object being wrapped.\n      grpc_debug_server_addresses: (`str` or `list` of `str`) Single or a list\n        of the gRPC debug server addresses, in the format of\n        <host:port>, with or without the "grpc://" prefix. For example:\n          "localhost:7000",\n          ["localhost:7000", "192.168.0.2:8000"]\n      watch_fn: (`Callable`) A Callable that can be used to define per-run\n        debug ops and watched tensors. See the doc of\n        `NonInteractiveDebugWrapperSession.__init__()` for details.\n      thread_name_filter: Regular-expression white list for threads on which the\n        wrapper session will be active. See doc of `BaseDebugWrapperSession` for\n        more details.\n\n    Raises:\n       TypeError: If `grpc_debug_server_addresses` is not a `str` or a `list`\n         of `str`.\n    '
        framework.NonInteractiveDebugWrapperSession.__init__(self, sess, watch_fn=watch_fn, thread_name_filter=thread_name_filter)
        if isinstance(grpc_debug_server_addresses, str):
            self._grpc_debug_server_urls = [self._normalize_grpc_url(grpc_debug_server_addresses)]
        elif isinstance(grpc_debug_server_addresses, list):
            self._grpc_debug_server_urls = []
            for address in grpc_debug_server_addresses:
                if not isinstance(address, str):
                    raise TypeError('Expected type str in list grpc_debug_server_addresses, received type %s' % type(address))
                self._grpc_debug_server_urls.append(self._normalize_grpc_url(address))
        else:
            raise TypeError('Expected type str or list in grpc_debug_server_addresses, received type %s' % type(grpc_debug_server_addresses))

    def prepare_run_debug_urls(self, fetches, feed_dict):
        if False:
            while True:
                i = 10
        'Implementation of abstract method in superclass.\n\n    See doc of `NonInteractiveDebugWrapperSession.prepare_run_debug_urls()`\n    for details.\n\n    Args:\n      fetches: Same as the `fetches` argument to `Session.run()`\n      feed_dict: Same as the `feed_dict` argument to `Session.run()`\n\n    Returns:\n      debug_urls: (`str` or `list` of `str`) file:// debug URLs to be used in\n        this `Session.run()` call.\n    '
        return self._grpc_debug_server_urls

    def _normalize_grpc_url(self, address):
        if False:
            i = 10
            return i + 15
        return common.GRPC_URL_PREFIX + address if not address.startswith(common.GRPC_URL_PREFIX) else address

def _signal_handler(unused_signal, unused_frame):
    if False:
        print('Hello World!')
    while True:
        response = input('\nSIGINT received. Quit program? (Y/n): ').strip()
        if response in ('', 'Y', 'y'):
            sys.exit(0)
        elif response in ('N', 'n'):
            break

def register_signal_handler():
    if False:
        while True:
            i = 10
    try:
        signal.signal(signal.SIGINT, _signal_handler)
    except ValueError:
        pass

class TensorBoardDebugWrapperSession(GrpcDebugWrapperSession):
    """A tfdbg Session wrapper that can be used with TensorBoard Debugger Plugin.

  This wrapper is the same as `GrpcDebugWrapperSession`, except that it uses a
    predefined `watch_fn` that
    1) uses `DebugIdentity` debug ops with the `gated_grpc` attribute set to
        `True` to allow the interactive enabling and disabling of tensor
       breakpoints.
    2) watches all tensors in the graph.
  This saves the need for the user to define a `watch_fn`.
  """

    def __init__(self, sess, grpc_debug_server_addresses, thread_name_filter=None, send_traceback_and_source_code=True):
        if False:
            print('Hello World!')
        'Constructor of TensorBoardDebugWrapperSession.\n\n    Args:\n      sess: The `tf.compat.v1.Session` instance to be wrapped.\n      grpc_debug_server_addresses: gRPC address(es) of debug server(s), as a\n        `str` or a `list` of `str`s. E.g., "localhost:2333",\n        "grpc://localhost:2333", ["192.168.0.7:2333", "192.168.0.8:2333"].\n      thread_name_filter: Optional filter for thread names.\n      send_traceback_and_source_code: Whether traceback of graph elements and\n        the source code are to be sent to the debug server(s).\n    '

        def _gated_grpc_watch_fn(fetches, feeds):
            if False:
                i = 10
                return i + 15
            del fetches, feeds
            return framework.WatchOptions(debug_ops=['DebugIdentity(gated_grpc=true)'])
        super().__init__(sess, grpc_debug_server_addresses, watch_fn=_gated_grpc_watch_fn, thread_name_filter=thread_name_filter)
        self._send_traceback_and_source_code = send_traceback_and_source_code
        self._sent_graph_version = -1
        register_signal_handler()

    def run(self, fetches, feed_dict=None, options=None, run_metadata=None, callable_runner=None, callable_runner_args=None, callable_options=None):
        if False:
            print('Hello World!')
        if self._send_traceback_and_source_code:
            self._sent_graph_version = publish_traceback(self._grpc_debug_server_urls, self.graph, feed_dict, fetches, self._sent_graph_version)
        return super().run(fetches, feed_dict=feed_dict, options=options, run_metadata=run_metadata, callable_runner=callable_runner, callable_runner_args=callable_runner_args, callable_options=callable_options)