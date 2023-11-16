"""Writer class for `DebugEvent` protos in tfdbg v2."""
import time
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.client import _pywrap_debug_events_writer
DEFAULT_CIRCULAR_BUFFER_SIZE = 1000

class DebugEventsWriter(object):
    """A writer for TF debugging events. Used by tfdbg v2."""

    def __init__(self, dump_root, tfdbg_run_id, circular_buffer_size=DEFAULT_CIRCULAR_BUFFER_SIZE):
        if False:
            for i in range(10):
                print('nop')
        'Construct a DebugEventsWriter object.\n\n    NOTE: Given the same `dump_root`, all objects from this constructor\n      will point to the same underlying set of writers. In other words, they\n      will write to the same set of debug events files in the `dump_root`\n      folder.\n\n    Args:\n      dump_root: The root directory for dumping debug data. If `dump_root` does\n        not exist as a directory, it will be created.\n      tfdbg_run_id: Debugger Run ID.\n      circular_buffer_size: Size of the circular buffer for each of the two\n        execution-related debug events files: with the following suffixes: -\n          .execution - .graph_execution_traces If <= 0, the circular-buffer\n          behavior will be abolished in the constructed object.\n    '
        if not dump_root:
            raise ValueError('Empty or None dump root')
        self._dump_root = dump_root
        self._tfdbg_run_id = tfdbg_run_id
        _pywrap_debug_events_writer.Init(self._dump_root, self._tfdbg_run_id, circular_buffer_size)

    def WriteSourceFile(self, source_file):
        if False:
            i = 10
            return i + 15
        'Write a SourceFile proto with the writer.\n\n    Args:\n      source_file: A SourceFile proto, describing the content of a source file\n        involved in the execution of the debugged TensorFlow program.\n    '
        debug_event = debug_event_pb2.DebugEvent(source_file=source_file)
        self._EnsureTimestampAdded(debug_event)
        _pywrap_debug_events_writer.WriteSourceFile(self._dump_root, debug_event)

    def WriteStackFrameWithId(self, stack_frame_with_id):
        if False:
            while True:
                i = 10
        'Write a StackFrameWithId proto with the writer.\n\n    Args:\n      stack_frame_with_id: A StackFrameWithId proto, describing the content a\n        stack frame involved in the execution of the debugged TensorFlow\n        program.\n    '
        debug_event = debug_event_pb2.DebugEvent(stack_frame_with_id=stack_frame_with_id)
        self._EnsureTimestampAdded(debug_event)
        _pywrap_debug_events_writer.WriteStackFrameWithId(self._dump_root, debug_event)

    def WriteGraphOpCreation(self, graph_op_creation):
        if False:
            while True:
                i = 10
        'Write a GraphOpCreation proto with the writer.\n\n    Args:\n      graph_op_creation: A GraphOpCreation proto, describing the details of the\n        creation of an op inside a TensorFlow Graph.\n    '
        debug_event = debug_event_pb2.DebugEvent(graph_op_creation=graph_op_creation)
        self._EnsureTimestampAdded(debug_event)
        _pywrap_debug_events_writer.WriteGraphOpCreation(self._dump_root, debug_event)

    def WriteDebuggedGraph(self, debugged_graph):
        if False:
            print('Hello World!')
        'Write a DebuggedGraph proto with the writer.\n\n    Args:\n      debugged_graph: A DebuggedGraph proto, describing the details of a\n        TensorFlow Graph that has completed its construction.\n    '
        debug_event = debug_event_pb2.DebugEvent(debugged_graph=debugged_graph)
        self._EnsureTimestampAdded(debug_event)
        _pywrap_debug_events_writer.WriteDebuggedGraph(self._dump_root, debug_event)

    def WriteExecution(self, execution):
        if False:
            return 10
        'Write a Execution proto with the writer.\n\n    Args:\n      execution: An Execution proto, describing a TensorFlow op or graph\n        execution event.\n    '
        debug_event = debug_event_pb2.DebugEvent(execution=execution)
        self._EnsureTimestampAdded(debug_event)
        _pywrap_debug_events_writer.WriteExecution(self._dump_root, debug_event)

    def WriteGraphExecutionTrace(self, graph_execution_trace):
        if False:
            return 10
        "Write a GraphExecutionTrace proto with the writer.\n\n    Args:\n      graph_execution_trace: A GraphExecutionTrace proto, concerning the value\n        of an intermediate tensor or a list of intermediate tensors that are\n        computed during the graph's execution.\n    "
        debug_event = debug_event_pb2.DebugEvent(graph_execution_trace=graph_execution_trace)
        self._EnsureTimestampAdded(debug_event)
        _pywrap_debug_events_writer.WriteGraphExecutionTrace(self._dump_root, debug_event)

    def RegisterDeviceAndGetId(self, device_name):
        if False:
            i = 10
            return i + 15
        return _pywrap_debug_events_writer.RegisterDeviceAndGetId(self._dump_root, device_name)

    def FlushNonExecutionFiles(self):
        if False:
            while True:
                i = 10
        'Flush the non-execution debug event files.'
        _pywrap_debug_events_writer.FlushNonExecutionFiles(self._dump_root)

    def FlushExecutionFiles(self):
        if False:
            i = 10
            return i + 15
        'Flush the execution debug event files.\n\n    Causes the current content of the cyclic buffers to be written to\n    the .execution and .graph_execution_traces debug events files.\n    Also clears those cyclic buffers.\n    '
        _pywrap_debug_events_writer.FlushExecutionFiles(self._dump_root)

    def Close(self):
        if False:
            return 10
        'Close the writer.'
        _pywrap_debug_events_writer.Close(self._dump_root)

    @property
    def dump_root(self):
        if False:
            return 10
        return self._dump_root

    def _EnsureTimestampAdded(self, debug_event):
        if False:
            for i in range(10):
                print('nop')
        if debug_event.wall_time == 0:
            debug_event.wall_time = time.time()