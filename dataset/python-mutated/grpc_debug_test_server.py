"""GRPC debug server for testing."""
import collections
import errno
import functools
import hashlib
import json
import os
import re
import tempfile
import threading
import time
import portpicker
from tensorflow.core.debug import debug_service_pb2
from tensorflow.core.protobuf import config_pb2
from tensorflow.core.util import event_pb2
from tensorflow.python.client import session
from tensorflow.python.debug.lib import debug_data
from tensorflow.python.debug.lib import debug_utils
from tensorflow.python.debug.lib import grpc_debug_server
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import errors
from tensorflow.python.lib.io import file_io
from tensorflow.python.ops import variables
from tensorflow.python.util import compat

def _get_dump_file_path(dump_root, device_name, debug_node_name):
    if False:
        for i in range(10):
            print('nop')
    'Get the file path of the dump file for a debug node.\n\n  Args:\n    dump_root: (str) Root dump directory.\n    device_name: (str) Name of the device that the debug node resides on.\n    debug_node_name: (str) Name of the debug node, e.g.,\n      cross_entropy/Log:0:DebugIdentity.\n\n  Returns:\n    (str) Full path of the dump file.\n  '
    dump_root = os.path.join(dump_root, debug_data.device_name_to_device_path(device_name))
    if '/' in debug_node_name:
        dump_dir = os.path.join(dump_root, os.path.dirname(debug_node_name))
        dump_file_name = re.sub(':', '_', os.path.basename(debug_node_name))
    else:
        dump_dir = dump_root
        dump_file_name = re.sub(':', '_', debug_node_name)
    now_microsec = int(round(time.time() * 1000 * 1000))
    dump_file_name += '_%d' % now_microsec
    return os.path.join(dump_dir, dump_file_name)

class EventListenerTestStreamHandler(grpc_debug_server.EventListenerBaseStreamHandler):
    """Implementation of EventListenerBaseStreamHandler that dumps to file."""

    def __init__(self, dump_dir, event_listener_servicer):
        if False:
            return 10
        super(EventListenerTestStreamHandler, self).__init__()
        self._dump_dir = dump_dir
        self._event_listener_servicer = event_listener_servicer
        if self._dump_dir:
            self._try_makedirs(self._dump_dir)
        self._grpc_path = None
        self._cached_graph_defs = []
        self._cached_graph_def_device_names = []
        self._cached_graph_def_wall_times = []

    def on_core_metadata_event(self, event):
        if False:
            i = 10
            return i + 15
        self._event_listener_servicer.toggle_watch()
        core_metadata = json.loads(event.log_message.message)
        if not self._grpc_path:
            grpc_path = core_metadata['grpc_path']
            if grpc_path:
                if grpc_path.startswith('/'):
                    grpc_path = grpc_path[1:]
            if self._dump_dir:
                self._dump_dir = os.path.join(self._dump_dir, grpc_path)
                for (graph_def, device_name, wall_time) in zip(self._cached_graph_defs, self._cached_graph_def_device_names, self._cached_graph_def_wall_times):
                    self._write_graph_def(graph_def, device_name, wall_time)
        if self._dump_dir:
            self._write_core_metadata_event(event)
        else:
            self._event_listener_servicer.core_metadata_json_strings.append(event.log_message.message)

    def on_graph_def(self, graph_def, device_name, wall_time):
        if False:
            while True:
                i = 10
        'Implementation of the tensor value-carrying Event proto callback.\n\n    Args:\n      graph_def: A GraphDef object.\n      device_name: Name of the device on which the graph was created.\n      wall_time: An epoch timestamp (in microseconds) for the graph.\n    '
        if self._dump_dir:
            if self._grpc_path:
                self._write_graph_def(graph_def, device_name, wall_time)
            else:
                self._cached_graph_defs.append(graph_def)
                self._cached_graph_def_device_names.append(device_name)
                self._cached_graph_def_wall_times.append(wall_time)
        else:
            self._event_listener_servicer.partition_graph_defs.append(graph_def)

    def on_value_event(self, event):
        if False:
            print('Hello World!')
        'Implementation of the tensor value-carrying Event proto callback.\n\n    Writes the Event proto to the file system for testing. The path written to\n    follows the same pattern as the file:// debug URLs of tfdbg, i.e., the\n    name scope of the op becomes the directory structure under the dump root\n    directory.\n\n    Args:\n      event: The Event proto carrying a tensor value.\n\n    Returns:\n      If the debug node belongs to the set of currently activated breakpoints,\n      a `EventReply` proto will be returned.\n    '
        if self._dump_dir:
            self._write_value_event(event)
        else:
            value = event.summary.value[0]
            tensor_value = debug_data.load_tensor_from_event(event)
            self._event_listener_servicer.debug_tensor_values[value.node_name].append(tensor_value)
            items = event.summary.value[0].node_name.split(':')
            node_name = items[0]
            output_slot = int(items[1])
            debug_op = items[2]
            if (node_name, output_slot, debug_op) in self._event_listener_servicer.breakpoints:
                return debug_service_pb2.EventReply()

    def _try_makedirs(self, dir_path):
        if False:
            for i in range(10):
                print('nop')
        if not os.path.isdir(dir_path):
            try:
                os.makedirs(dir_path)
            except OSError as error:
                if error.errno != errno.EEXIST:
                    raise

    def _write_core_metadata_event(self, event):
        if False:
            i = 10
            return i + 15
        core_metadata_path = os.path.join(self._dump_dir, debug_data.METADATA_FILE_PREFIX + debug_data.CORE_METADATA_TAG + '_%d' % event.wall_time)
        self._try_makedirs(self._dump_dir)
        with open(core_metadata_path, 'wb') as f:
            f.write(event.SerializeToString())

    def _write_graph_def(self, graph_def, device_name, wall_time):
        if False:
            while True:
                i = 10
        encoded_graph_def = graph_def.SerializeToString()
        graph_hash = int(hashlib.sha1(encoded_graph_def).hexdigest(), 16)
        event = event_pb2.Event(graph_def=encoded_graph_def, wall_time=wall_time)
        graph_file_path = os.path.join(self._dump_dir, debug_data.device_name_to_device_path(device_name), debug_data.METADATA_FILE_PREFIX + debug_data.GRAPH_FILE_TAG + debug_data.HASH_TAG + '%d_%d' % (graph_hash, wall_time))
        self._try_makedirs(os.path.dirname(graph_file_path))
        with open(graph_file_path, 'wb') as f:
            f.write(event.SerializeToString())

    def _write_value_event(self, event):
        if False:
            while True:
                i = 10
        value = event.summary.value[0]
        summary_metadata = event.summary.value[0].metadata
        if not summary_metadata.plugin_data:
            raise ValueError('The value lacks plugin data.')
        try:
            content = json.loads(compat.as_text(summary_metadata.plugin_data.content))
        except ValueError as err:
            raise ValueError('Could not parse content into JSON: %r, %r' % (content, err))
        device_name = content['device']
        dump_full_path = _get_dump_file_path(self._dump_dir, device_name, value.node_name)
        self._try_makedirs(os.path.dirname(dump_full_path))
        with open(dump_full_path, 'wb') as f:
            f.write(event.SerializeToString())

class EventListenerTestServicer(grpc_debug_server.EventListenerBaseServicer):
    """An implementation of EventListenerBaseServicer for testing."""

    def __init__(self, server_port, dump_dir, toggle_watch_on_core_metadata=None):
        if False:
            while True:
                i = 10
        'Constructor of EventListenerTestServicer.\n\n    Args:\n      server_port: (int) The server port number.\n      dump_dir: (str) The root directory to which the data files will be\n        dumped. If empty or None, the received debug data will not be dumped\n        to the file system: they will be stored in memory instead.\n      toggle_watch_on_core_metadata: A list of\n        (node_name, output_slot, debug_op) tuples to toggle the\n        watchpoint status during the on_core_metadata calls (optional).\n    '
        self.core_metadata_json_strings = []
        self.partition_graph_defs = []
        self.debug_tensor_values = collections.defaultdict(list)
        self._initialize_toggle_watch_state(toggle_watch_on_core_metadata)
        grpc_debug_server.EventListenerBaseServicer.__init__(self, server_port, functools.partial(EventListenerTestStreamHandler, dump_dir, self))
        self._call_types = []
        self._call_keys = []
        self._origin_stacks = []
        self._origin_id_to_strings = []
        self._graph_tracebacks = []
        self._graph_versions = []
        self._source_files = []

    def _initialize_toggle_watch_state(self, toggle_watches):
        if False:
            i = 10
            return i + 15
        self._toggle_watches = toggle_watches
        self._toggle_watch_state = {}
        if self._toggle_watches:
            for watch_key in self._toggle_watches:
                self._toggle_watch_state[watch_key] = False

    def toggle_watch(self):
        if False:
            while True:
                i = 10
        for watch_key in self._toggle_watch_state:
            (node_name, output_slot, debug_op) = watch_key
            if self._toggle_watch_state[watch_key]:
                self.request_unwatch(node_name, output_slot, debug_op)
            else:
                self.request_watch(node_name, output_slot, debug_op)
            self._toggle_watch_state[watch_key] = not self._toggle_watch_state[watch_key]

    def clear_data(self):
        if False:
            return 10
        self.core_metadata_json_strings = []
        self.partition_graph_defs = []
        self.debug_tensor_values = collections.defaultdict(list)
        self._call_types = []
        self._call_keys = []
        self._origin_stacks = []
        self._origin_id_to_strings = []
        self._graph_tracebacks = []
        self._graph_versions = []
        self._source_files = []

    def SendTracebacks(self, request, context):
        if False:
            for i in range(10):
                print('nop')
        self._call_types.append(request.call_type)
        self._call_keys.append(request.call_key)
        self._origin_stacks.append(request.origin_stack)
        self._origin_id_to_strings.append(request.origin_id_to_string)
        self._graph_tracebacks.append(request.graph_traceback)
        self._graph_versions.append(request.graph_version)
        return debug_service_pb2.EventReply()

    def SendSourceFiles(self, request, context):
        if False:
            while True:
                i = 10
        self._source_files.append(request)
        return debug_service_pb2.EventReply()

    def query_op_traceback(self, op_name):
        if False:
            for i in range(10):
                print('nop')
        'Query the traceback of an op.\n\n    Args:\n      op_name: Name of the op to query.\n\n    Returns:\n      The traceback of the op, as a list of 3-tuples:\n        (filename, lineno, function_name)\n\n    Raises:\n      ValueError: If the op cannot be found in the tracebacks received by the\n        server so far.\n    '
        for op_log_proto in self._graph_tracebacks:
            for log_entry in op_log_proto.log_entries:
                if log_entry.name == op_name:
                    return self._code_def_to_traceback(log_entry.code_def, op_log_proto.id_to_string)
        raise ValueError("Op '%s' does not exist in the tracebacks received by the debug server." % op_name)

    def query_origin_stack(self):
        if False:
            print('Hello World!')
        'Query the stack of the origin of the execution call.\n\n    Returns:\n      A `list` of all tracebacks. Each item corresponds to an execution call,\n        i.e., a `SendTracebacks` request. Each item is a `list` of 3-tuples:\n        (filename, lineno, function_name).\n    '
        ret = []
        for (stack, id_to_string) in zip(self._origin_stacks, self._origin_id_to_strings):
            ret.append(self._code_def_to_traceback(stack, id_to_string))
        return ret

    def query_call_types(self):
        if False:
            i = 10
            return i + 15
        return self._call_types

    def query_call_keys(self):
        if False:
            return 10
        return self._call_keys

    def query_graph_versions(self):
        if False:
            return 10
        return self._graph_versions

    def query_source_file_line(self, file_path, lineno):
        if False:
            print('Hello World!')
        'Query the content of a given line in a source file.\n\n    Args:\n      file_path: Path to the source file.\n      lineno: Line number as an `int`.\n\n    Returns:\n      Content of the line as a string.\n\n    Raises:\n      ValueError: If no source file is found at the given file_path.\n    '
        if not self._source_files:
            raise ValueError('This debug server has not received any source file contents yet.')
        for source_files in self._source_files:
            for source_file_proto in source_files.source_files:
                if source_file_proto.file_path == file_path:
                    return source_file_proto.lines[lineno - 1]
        raise ValueError('Source file at path %s has not been received by the debug server', file_path)

    def _code_def_to_traceback(self, code_def, id_to_string):
        if False:
            i = 10
            return i + 15
        return [(id_to_string[trace.file_id], trace.lineno, id_to_string[trace.function_id]) for trace in code_def.traces]

def start_server_on_separate_thread(dump_to_filesystem=True, server_start_delay_sec=0.0, poll_server=False, blocking=True, toggle_watch_on_core_metadata=None):
    if False:
        i = 10
        return i + 15
    "Create a test gRPC debug server and run on a separate thread.\n\n  Args:\n    dump_to_filesystem: (bool) whether the debug server will dump debug data\n      to the filesystem.\n    server_start_delay_sec: (float) amount of time (in sec) to delay the server\n      start up for.\n    poll_server: (bool) whether the server will be polled till success on\n      startup.\n    blocking: (bool) whether the server should be started in a blocking mode.\n    toggle_watch_on_core_metadata: A list of\n        (node_name, output_slot, debug_op) tuples to toggle the\n        watchpoint status during the on_core_metadata calls (optional).\n\n  Returns:\n    server_port: (int) Port on which the server runs.\n    debug_server_url: (str) grpc:// URL to the server.\n    server_dump_dir: (str) The debug server's dump directory.\n    server_thread: The server Thread object.\n    server: The `EventListenerTestServicer` object.\n\n  Raises:\n    ValueError: If polling the server process for ready state is not successful\n      within maximum polling count.\n  "
    server_port = portpicker.pick_unused_port()
    debug_server_url = 'grpc://localhost:%d' % server_port
    server_dump_dir = tempfile.mkdtemp() if dump_to_filesystem else None
    server = EventListenerTestServicer(server_port=server_port, dump_dir=server_dump_dir, toggle_watch_on_core_metadata=toggle_watch_on_core_metadata)

    def delay_then_run_server():
        if False:
            while True:
                i = 10
        time.sleep(server_start_delay_sec)
        server.run_server(blocking=blocking)
    server_thread = threading.Thread(target=delay_then_run_server)
    server_thread.start()
    if poll_server:
        if not _poll_server_till_success(50, 0.2, debug_server_url, server_dump_dir, server, gpu_memory_fraction=0.1):
            raise ValueError('Failed to start test gRPC debug server at port %d' % server_port)
        server.clear_data()
    return (server_port, debug_server_url, server_dump_dir, server_thread, server)

def _poll_server_till_success(max_attempts, sleep_per_poll_sec, debug_server_url, dump_dir, server, gpu_memory_fraction=1.0):
    if False:
        for i in range(10):
            print('nop')
    'Poll server until success or exceeding max polling count.\n\n  Args:\n    max_attempts: (int) How many times to poll at maximum\n    sleep_per_poll_sec: (float) How many seconds to sleep for after each\n      unsuccessful poll.\n    debug_server_url: (str) gRPC URL to the debug server.\n    dump_dir: (str) Dump directory to look for files in. If None, will directly\n      check data from the server object.\n    server: The server object.\n    gpu_memory_fraction: (float) Fraction of GPU memory to be\n      allocated for the Session used in server polling.\n\n  Returns:\n    (bool) Whether the polling succeeded within max_polls attempts.\n  '
    poll_count = 0
    config = config_pb2.ConfigProto(gpu_options=config_pb2.GPUOptions(per_process_gpu_memory_fraction=gpu_memory_fraction))
    with session.Session(config=config) as sess:
        for poll_count in range(max_attempts):
            server.clear_data()
            print('Polling: poll_count = %d' % poll_count)
            x_init_name = 'x_init_%d' % poll_count
            x_init = constant_op.constant([42.0], shape=[1], name=x_init_name)
            x = variables.Variable(x_init, name=x_init_name)
            run_options = config_pb2.RunOptions()
            debug_utils.add_debug_tensor_watch(run_options, x_init_name, 0, debug_urls=[debug_server_url])
            try:
                sess.run(x.initializer, options=run_options)
            except errors.FailedPreconditionError:
                pass
            if dump_dir:
                if os.path.isdir(dump_dir) and debug_data.DebugDumpDir(dump_dir).size > 0:
                    file_io.delete_recursively(dump_dir)
                    print('Poll succeeded.')
                    return True
                else:
                    print('Poll failed. Sleeping for %f s' % sleep_per_poll_sec)
                    time.sleep(sleep_per_poll_sec)
            elif server.debug_tensor_values:
                print('Poll succeeded.')
                return True
            else:
                print('Poll failed. Sleeping for %f s' % sleep_per_poll_sec)
                time.sleep(sleep_per_poll_sec)
        return False