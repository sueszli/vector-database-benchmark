"""Monitors for Debug Events in the tfdbg2 format.

Monitors get access to graph-building- and execution-related data
objects as the DebugDataReader (see `debug_events_reader.py`) reads the
data in a continuous fashion, via a set of callbacks. This mechanism enables
hooking custom logic into the DebugEvent reading stream without the need for
any polling or iterating over the entire data held by DebugDataReader.

This module includes the following built-in hooks:
  - InfNanMonitor: Monitors infinity and nan values in top-level execution and
    intra-graph execution events.

When a monitor (subtype of `BaseMonitor`) is constructed with a DebugDataReader
as the first argument of the constructor call, the monitor is automatically
registered with the DebugDataReader. For example:

```py
debug_data_reader = debug_events_reader.DebugDataReader(dump_dir)
inf_nan_monitor = debug_events_monitors.InfNanMonitor(debug_data_reader)

debug_data_reader.update()
# `inf_nan_monitor`'s on_* methods will get called as the execution-related
# and other types of data are read by `debug_data_reader`.
```
"""
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2

class BaseMonitor(object):
    """Base class for debug event data monitors."""

    def __init__(self, debug_events_reader):
        if False:
            return 10
        self._debug_data_reader = debug_events_reader
        debug_events_reader._add_monitor(self)

    def on_execution(self, execution_index, execution):
        if False:
            for i in range(10):
                print('nop')
        'Monitor method for top-level execution events.\n\n    Return values (if any) are ignored by the associated DebugDataReader.\n\n    Args:\n      execution_index: The index of the top-level execution event, as an int.\n      execution: An Execution data object, for a top-level op or function\n        execution event.\n    '

    def on_graph_execution_trace(self, graph_execution_trace_index, graph_execution_trace):
        if False:
            for i in range(10):
                print('nop')
        'Monitor method for intra-graph execution events.\n\n    Return values (if any) are ignored by the associated DebugDataReader.\n\n    Args:\n      graph_execution_trace_index: The index of the intra-graph execution\n        event, as an int.\n      graph_execution_trace: A GraphExecutionTrace data object, for an\n        intra-graph tensor event.\n    '

class InfNanAlert(object):
    """Alert for Infinity and NaN values."""

    def __init__(self, wall_time, op_type, output_slot, size=None, num_neg_inf=None, num_pos_inf=None, num_nan=None, execution_index=None, graph_execution_trace_index=None):
        if False:
            i = 10
            return i + 15
        self._wall_time = wall_time
        self._op_type = op_type
        self._output_slot = output_slot
        self._size = size
        self._num_neg_inf = num_neg_inf
        self._num_pos_inf = num_pos_inf
        self._num_nan = num_nan
        self._execution_index = execution_index
        self._graph_execution_trace_index = graph_execution_trace_index

    @property
    def wall_time(self):
        if False:
            return 10
        return self._wall_time

    @property
    def op_type(self):
        if False:
            return 10
        return self._op_type

    @property
    def output_slot(self):
        if False:
            for i in range(10):
                print('nop')
        return self._output_slot

    @property
    def size(self):
        if False:
            while True:
                i = 10
        return self._size

    @property
    def num_neg_inf(self):
        if False:
            while True:
                i = 10
        return self._num_neg_inf

    @property
    def num_pos_inf(self):
        if False:
            return 10
        return self._num_pos_inf

    @property
    def num_nan(self):
        if False:
            print('Hello World!')
        return self._num_nan

    @property
    def execution_index(self):
        if False:
            i = 10
            return i + 15
        return self._execution_index

    @property
    def graph_execution_trace_index(self):
        if False:
            for i in range(10):
                print('nop')
        return self._graph_execution_trace_index

class InfNanMonitor(BaseMonitor):
    """Monitor for Infinity and NaN in tensor values."""

    def __init__(self, debug_events_reader, limit=0):
        if False:
            return 10
        super(InfNanMonitor, self).__init__(debug_events_reader)
        self._limit = limit
        self._alerts = []

    def _check_full_tensor_value(self, tensor_value, wall_time, op_type, output_slot, execution_index=None, graph_execution_trace_index=None):
        if False:
            print('Hello World!')
        'Check a full tensor value.\n\n    Appends to the list of alerts if any inf or nan is found in the full tensor\n    value.\n\n    Args:\n      tensor_value: The full tensor value as a `np.ndarray`.\n      wall_time: Wall timestamp for the execution event that generated the\n        tensor value.\n      op_type: Op type executed.\n      output_slot: The output slot of the op.\n      execution_index: Index to the top-level execution event.\n      graph_execution_trace_index: Index to the intra-graph execution trace\n        (if applicable.)\n    '
        size = np.size(tensor_value)
        if not size or not np.issubdtype(tensor_value.dtype, np.floating):
            return
        is_inf = np.isinf(tensor_value)
        num_neg_inf = np.count_nonzero(np.logical_and(is_inf, np.less(tensor_value, 0.0)))
        num_pos_inf = np.count_nonzero(np.logical_and(is_inf, np.greater(tensor_value, 0.0)))
        num_nan = np.count_nonzero(np.isnan(tensor_value))
        if num_neg_inf or num_pos_inf or num_nan:
            self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))

    def _check_debug_tensor_value(self, tensor_debug_mode, debug_tensor_value, wall_time, op_type, output_slot, execution_index=None, graph_execution_trace_index=None):
        if False:
            for i in range(10):
                print('nop')
        'Check for bad numerical values based on debug summary of tensor value.\n\n    If tensor_debug_mode is one in which debug_tensor_value does not carry\n    information about the presence or count of inf / nan values (e.g., SHAPE),\n    this method is a no-op.\n\n    When infs and/or nans are found, `InfNanAlert` objects are created and\n    appended to `self._alerts`.\n\n    Args:\n      tensor_debug_mode: TensorDebugMode proto enum.\n      debug_tensor_value: Debug tensor value as a list of numbers.\n      wall_time: Wall timestamp for the tensor event.\n      op_type: Type of the op that generated the tensor (e.g., "Conv2D").\n      output_slot: Output slot index of the tensor for the op.\n      execution_index: Top-level execution index.\n      graph_execution_trace_index: Intra-graph execution index.\n    '
        assert tensor_debug_mode != debug_event_pb2.TensorDebugMode.FULL_TENSOR
        if not debug_tensor_value:
            return
        if tensor_debug_mode == debug_event_pb2.TensorDebugMode.CURT_HEALTH:
            (_, any_nan_inf) = debug_tensor_value
            if any_nan_inf:
                self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))
        elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.CONCISE_HEALTH:
            (_, size, num_neg_inf, num_pos_inf, num_nan) = debug_tensor_value
            if num_neg_inf or num_pos_inf or num_nan:
                self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))
        elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_HEALTH:
            (_, _, _, _, size, num_neg_inf, num_pos_inf, num_nan, _, _, _) = debug_tensor_value
            if num_neg_inf or num_pos_inf or num_nan:
                self._alerts.append(InfNanAlert(wall_time, op_type, output_slot, size=size, num_neg_inf=num_neg_inf, num_pos_inf=num_pos_inf, num_nan=num_nan, execution_index=execution_index, graph_execution_trace_index=graph_execution_trace_index))

    def on_execution(self, execution_index, execution):
        if False:
            return 10
        if self._limit > 0 and len(self._alerts) >= self._limit:
            return
        if execution.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
            tensor_values = self._debug_data_reader.execution_to_tensor_values(execution)
            for (output_slot, tensor_value) in enumerate(tensor_values):
                self._check_full_tensor_value(tensor_value, execution.wall_time, execution.op_type, output_slot, execution_index=execution_index)
        elif execution.debug_tensor_values:
            for (output_slot, debug_tensor_value) in enumerate(execution.debug_tensor_values):
                self._check_debug_tensor_value(execution.tensor_debug_mode, debug_tensor_value, execution.wall_time, execution.op_type, output_slot, execution_index=execution_index)

    def on_graph_execution_trace(self, graph_execution_trace_index, graph_execution_trace):
        if False:
            return 10
        'Monitor method for GraphExecutionTrace data object.'
        if self._limit > 0 and len(self._alerts) >= self._limit:
            return
        if graph_execution_trace.tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
            tensor_value = self._debug_data_reader.graph_execution_trace_to_tensor_value(graph_execution_trace)
            self._check_full_tensor_value(tensor_value, graph_execution_trace.wall_time, graph_execution_trace.op_type, graph_execution_trace.output_slot, graph_execution_trace_index=graph_execution_trace_index)
        elif graph_execution_trace.debug_tensor_value:
            self._check_debug_tensor_value(graph_execution_trace.tensor_debug_mode, graph_execution_trace.debug_tensor_value, graph_execution_trace.wall_time, graph_execution_trace.op_type, graph_execution_trace.output_slot, graph_execution_trace_index=graph_execution_trace_index)

    def alerts(self):
        if False:
            i = 10
            return i + 15
        return self._alerts