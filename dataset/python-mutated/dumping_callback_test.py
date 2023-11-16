"""Unit tests for tfdbg v2 dumping callback."""
import collections
import os
import shutil
import socket
import tempfile
import threading
from absl.testing import parameterized
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_reader
from tensorflow.python.debug.lib import dumping_callback
from tensorflow.python.debug.lib import dumping_callback_test_lib
from tensorflow.python.eager import context
from tensorflow.python.eager import def_function
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import test_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import variables
from tensorflow.python.platform import googletest
from tensorflow.python.platform import test
from tensorflow.python.platform import tf_logging
_host_name = socket.gethostname()
_current_file_full_path = os.path.abspath(__file__)

class DumpingCallbackTest(dumping_callback_test_lib.DumpingCallbackTestBase, parameterized.TestCase):

    def setUp(self):
        if False:
            return 10
        super(DumpingCallbackTest, self).setUp()
        self.dump_root = tempfile.mkdtemp()

    def tearDown(self):
        if False:
            i = 10
            return i + 15
        if os.path.isdir(self.dump_root):
            shutil.rmtree(self.dump_root, ignore_errors=True)
        dumping_callback.disable_dump_debug_info()
        super(DumpingCallbackTest, self).tearDown()

    def _verifyStackFrames(self, stack_frames):
        if False:
            i = 10
            return i + 15
        'Verify the correctness of the stack frames.\n\n    Currently, it simply asserts that the current file is found in the stack\n    frames.\n    TODO(cais): Perhaps implement a stricter check later.\n\n    Args:\n      stack_frames: The stack frames to verify.\n    '
        self.assertTrue([frame for frame in stack_frames if frame[0] == _current_file_full_path])

    def _expectedDefaultDeviceName(self):
        if False:
            i = 10
            return i + 15
        gpu_name = test_util.gpu_device_name()
        if gpu_name:
            return '/job:localhost/replica:0/task:0' + gpu_name
        else:
            return '/job:localhost/replica:0/task:0/device:CPU:0'

    def testInvalidTensorDebugModeCausesError(self):
        if False:
            for i in range(10):
                print('nop')
        with self.assertRaisesRegex(ValueError, "Invalid value in tensor_debug_mode \\(\\'NONSENSICAL\\'\\).*Valid options.*NO_TENSOR.*"):
            dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='NONSENSICAL')

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('CurtHealth', 'CURT_HEALTH'), ('ConciseHealth', 'CONCISE_HEALTH'), ('Shape', 'SHAPE'), ('FulHealth', 'FULL_HEALTH'), ('FullTensor', 'FULL_TENSOR'))
    def testEnableDumpDebugInfoLogsTensorDebugModeAsStringName(self, tensor_debug_mode):
        if False:
            while True:
                i = 10
        log_messages = []

        def fake_logging_info(*args):
            if False:
                i = 10
                return i + 15
            log_messages.append(args)
        with test.mock.patch.object(tf_logging, 'info', side_effect=fake_logging_info):
            dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)
            self.assertLen(log_messages, 1)
            self.assertIn(self.dump_root, log_messages[0])
            self.assertIn(tensor_debug_mode, log_messages[0])

    def testDisablingTracingCallbackWithoutEnablingFirstIsTolerated(self):
        if False:
            i = 10
            return i + 15
        dumping_callback.disable_dump_debug_info()

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('CurtHealth', 'CURT_HEALTH'), ('ConciseHealth', 'CONCISE_HEALTH'), ('Shape', 'SHAPE'), ('FullHealth', 'FULL_HEALTH'), ('FullTensor', 'FULL_TENSOR'))
    def testPureEagerOpExecution(self, tensor_debug_mode):
        if False:
            i = 10
            return i + 15
        'Test dumping data from eager op execution: float32.'
        x = constant_op.constant(10.0)
        zero = constant_op.constant(0.0)
        one = constant_op.constant(1.0)
        two = constant_op.constant(2.0)
        three = constant_op.constant(3.0)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)
        while x > one:
            if math_ops.equal(x % two, zero):
                x = x / two
            else:
                x = x * three + one
        writer.FlushNonExecutionFiles()
        self._readAndCheckMetadataFile()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            self.assertFalse(reader.executions())
            writer.FlushExecutionFiles()
            reader.update()
            executions = reader.executions()
            prev_wall_time = 1
            executed_op_types = []
            tensor_values = collections.defaultdict(lambda : [])
            for execution in executions:
                self.assertGreaterEqual(execution.wall_time, prev_wall_time)
                prev_wall_time = execution.wall_time
                executed_op_types.append(execution.op_type)
                if execution.op_type in ('AddV2', 'Mul', 'RealDiv'):
                    self.assertLen(execution.output_tensor_device_ids, 1)
                    self.assertEqual(reader.device_name_by_id(execution.output_tensor_device_ids[0]), self._expectedDefaultDeviceName(), 'Unexpected device name from eager op %s' % execution.op_type)
                self.assertFalse(execution.graph_id)
                self.assertTrue(execution.input_tensor_ids)
                self.assertTrue(execution.output_tensor_ids)
                self.assertEqual(debug_event_pb2.TensorDebugMode.keys()[execution.tensor_debug_mode], tensor_debug_mode)
                if tensor_debug_mode == 'NO_TENSOR':
                    self.assertFalse(execution.debug_tensor_values)
                elif tensor_debug_mode == 'CURT_HEALTH':
                    self.assertLen(execution.debug_tensor_values, 1)
                    if execution.op_type in ('AddV2', 'Mul', 'RealDiv'):
                        self.assertAllClose(execution.debug_tensor_values, [[-1.0, 0.0]])
                elif tensor_debug_mode == 'CONCISE_HEALTH':
                    if execution.op_type in ('AddV2', 'Mul', 'RealDiv'):
                        self.assertAllClose(execution.debug_tensor_values, [[-1, 1, 0, 0, 0]])
                elif tensor_debug_mode == 'FULL_HEALTH':
                    if execution.op_type in ('AddV2', 'Mul', 'RealDiv'):
                        self.assertAllClose(execution.debug_tensor_values, [[-1, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1]])
                elif tensor_debug_mode == 'SHAPE':
                    if execution.op_type in ('AddV2', 'Mul', 'RealDiv'):
                        self.assertAllClose(execution.debug_tensor_values, [[-1, 1, 0, 1, 0, 0, 0, 0, 0, 0]])
                elif tensor_debug_mode == 'FULL_TENSOR':
                    tensor_values[execution.op_type].append(reader.execution_to_tensor_values(execution)[0])
                (host_name, stack_frames) = reader.read_execution_stack_trace(execution)
                self.assertEqual(host_name, _host_name)
                self._verifyStackFrames(stack_frames)
            if tensor_debug_mode == 'FULL_TENSOR':
                self.assertAllClose(tensor_values['Greater'], [1, 1, 1, 1, 1, 1, 0])
                self.assertAllClose(tensor_values['RealDiv'], [5, 8, 4, 2, 1])
                self.assertAllClose(tensor_values['Mul'], [15])
                self.assertAllClose(tensor_values['AddV2'], [16])
            self.assertEqual(executed_op_types, ['Greater', 'FloorMod', 'Equal', 'RealDiv', 'Greater', 'FloorMod', 'Equal', 'Mul', 'AddV2', 'Greater', 'FloorMod', 'Equal', 'RealDiv', 'Greater', 'FloorMod', 'Equal', 'RealDiv', 'Greater', 'FloorMod', 'Equal', 'RealDiv', 'Greater', 'FloorMod', 'Equal', 'RealDiv', 'Greater'])
            self.assertFalse(reader.outermost_graphs())
            self.assertEqual(reader.num_graph_execution_traces(), 0)

    @parameterized.named_parameters(('CurtHealth', 'CURT_HEALTH'), ('ConciseHealth', 'CONCISE_HEALTH'), ('FullHealth', 'FULL_HEALTH'), ('Shape', 'SHAPE'))
    @test_util.run_in_graph_and_eager_modes
    def testModesSummarizingBadNumericalValue(self, tensor_debug_mode):
        if False:
            print('Hello World!')
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)

        @def_function.function
        def func(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return (x + y) / (x - y)
        x = np.array([-3, -1, 0, 0, 1, 1, 1, 2], dtype=np.float16)
        y = np.array([2, -1, 0, 0, 1, 1, 1, 3], dtype=np.float16)
        self.evaluate(func(x, y))
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            graph_exec_traces = reader.graph_execution_traces()
            executed_op_types = [trace.op_type for trace in graph_exec_traces if trace.op_type not in ['Const', 'Placeholder']]
            self.assertCountEqual(executed_op_types, ['AddV2', 'Sub', 'RealDiv'])
            if tensor_debug_mode == 'CURT_HEALTH':
                for trace in graph_exec_traces:
                    tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                    self.assertGreaterEqual(tensor_id, 0)
                    if trace.op_type == 'RealDiv':
                        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 1])
                    else:
                        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0])
            elif tensor_debug_mode == 'CONCISE_HEALTH':
                for trace in graph_exec_traces:
                    tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                    self.assertGreaterEqual(tensor_id, 0)
                    if trace.op_type == 'RealDiv':
                        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 8, 1, 3, 2])
                    else:
                        self.assertAllClose(trace.debug_tensor_value, [tensor_id, 8, 0, 0, 0])
            elif tensor_debug_mode == 'FULL_HEALTH':
                for trace in graph_exec_traces:
                    tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                    self.assertGreaterEqual(tensor_id, 0)
                    if trace.op_type == 'RealDiv':
                        self.assertAllClose(trace.debug_tensor_value, [tensor_id, -1, 19, 1, 8, 1, 3, 2, 1, 0, 1])
                    elif trace.op_type == 'Sub':
                        self.assertAllClose(trace.debug_tensor_value, [tensor_id, -1, 19, 1, 8, 0, 0, 0, 2, 6, 0])
            else:
                for trace in graph_exec_traces:
                    tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                    self.assertGreaterEqual(tensor_id, 0)
                    self.assertAllClose(trace.debug_tensor_value, [tensor_id, 19, 1, 8, 8, 0, 0, 0, 0, 0])

    @parameterized.named_parameters(('CurtHealth', 'CURT_HEALTH'), ('FullTensor', 'FULL_TENSOR'))
    @test_util.run_in_graph_and_eager_modes
    def testConstTensorsAreCaptured(self, tensor_debug_mode):
        if False:
            return 10
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)

        @def_function.function
        def times_two_plus_three(x):
            if False:
                while True:
                    i = 10
            return x * constant_op.constant(2.0) + constant_op.constant(3.0)
        self.assertAllEqual(self.evaluate(times_two_plus_three(10.0)), 23.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            const_traces = [trace for trace in reader.graph_execution_traces() if trace.op_type == 'Const']
            self.assertGreaterEqual(len(const_traces), 3)
            if tensor_debug_mode == 'CURT_HEALTH':
                self.assertLen(const_traces[0].debug_tensor_value, 2)
                self.assertEqual(const_traces[0].debug_tensor_value[1], 0)
                self.assertLen(const_traces[1].debug_tensor_value, 2)
                self.assertEqual(const_traces[1].debug_tensor_value[1], 0)
                self.assertLen(const_traces[2].debug_tensor_value, 2)
                self.assertEqual(const_traces[2].debug_tensor_value[1], 0)
            else:
                const_tensor_values = [reader.graph_execution_trace_to_tensor_value(const_trace) for const_trace in const_traces]
                self.assertIn(10.0, const_tensor_values)
                self.assertIn(2.0, const_tensor_values)
                self.assertIn(3.0, const_tensor_values)

    @parameterized.named_parameters(('Shape', 'SHAPE'))
    @test_util.run_in_graph_and_eager_modes
    def testBooleanTensors(self, tensor_debug_mode):
        if False:
            while True:
                i = 10
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)

        @def_function.function
        def func(x, y):
            if False:
                while True:
                    i = 10
            return math_ops.logical_not(math_ops.logical_and(x, y))
        x = np.array([[False, False], [True, True]], dtype=np.bool_)
        y = np.array([[False, True], [False, True]], dtype=np.bool_)
        self.assertAllEqual(self.evaluate(func(x, y)), [[True, True], [True, False]])
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            graph_exec_traces = reader.graph_execution_traces()
            executed_op_types = [trace.op_type for trace in graph_exec_traces if trace.op_type not in ['Const', 'Placeholder']]
            self.assertEqual(executed_op_types, ['LogicalAnd', 'LogicalNot'])
            for trace in graph_exec_traces:
                tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                self.assertGreaterEqual(tensor_id, 0)
                self.assertAllClose(trace.debug_tensor_value, [tensor_id, 10, 2, 4, 2, 2, 0, 0, 0, 0])

    def testListingSourceFiles(self):
        if False:
            return 10
        writer = dumping_callback.enable_dump_debug_info(self.dump_root)
        self.assertAllClose(math_ops.truediv(7.0, 1.0 / 6.0), 42.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            source_file_list = reader.source_file_list()
            self.assertIsInstance(source_file_list, tuple)
            for item in source_file_list:
                self.assertIsInstance(item, tuple)
                self.assertLen(item, 2)
            self.assertIn((_host_name, _current_file_full_path), source_file_list)

    def testReadingSourceLines(self):
        if False:
            print('Hello World!')
        writer = dumping_callback.enable_dump_debug_info(self.dump_root)
        self.assertAllClose(math_ops.truediv(7.0, 1.0 / 6.0), 42.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            with open(_current_file_full_path, 'rt') as f:
                file_lines = f.read().split('\n')
            self.assertEqual(reader.source_lines(_host_name, _current_file_full_path), file_lines)

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('CurtHealth', 'CURT_HEALTH'), ('ConciseHealth', 'CONCISE_HEALTH'), ('FullHealth', 'FULL_HEALTH'), ('Shape', 'SHAPE'), ('FullTensor', 'FULL_TENSOR'))
    @test_util.run_in_graph_and_eager_modes
    def testNestedFunctionExecutionWithoutControlFlow(self, tensor_debug_mode):
        if False:
            print('Hello World!')
        x = constant_op.constant(2.0)
        y = constant_op.constant(3.0)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)

        @def_function.function
        def log_sum(x, y):
            if False:
                return 10
            return math_ops.log(x + y)

        @def_function.function
        def sin1p_log_sum(x, y):
            if False:
                return 10
            return math_ops.sin(1.0 + log_sum(x, y))
        self.assertAllClose(sin1p_log_sum(x, y), np.sin(1.0 + np.log(5.0)))
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            outermost_graphs = reader.outermost_graphs()
            self.assertLen(outermost_graphs, 1)
            if context.executing_eagerly():
                executions = reader.executions()
                self.assertLen(executions, 1)
                self.assertIn('sin1p_log_sum', executions[0].op_type)
                graph = reader.graph_by_id(executions[0].graph_id)
                self.assertEqual(graph.name, 'sin1p_log_sum')
                self.assertLen(graph.inner_graph_ids, 1)
                inner_graph = reader.graph_by_id(graph.inner_graph_ids[0])
                self.assertEqual(inner_graph.name, 'log_sum')
                self.assertLen(executions[0].output_tensor_device_ids, 1)
                self.assertEqual(reader.device_name_by_id(executions[0].output_tensor_device_ids[0]), self._expectedDefaultDeviceName())
                self.assertIn(self._expectedDefaultDeviceName(), set(reader.device_name_map().values()))
            add_op_digests = reader.graph_op_digests(op_type='AddV2')
            self.assertLen(add_op_digests, 2)
            self.assertEqual(reader.graph_by_id(add_op_digests[0].graph_id).name, 'log_sum')
            self.assertEqual(reader.graph_by_id(add_op_digests[1].graph_id).name, 'sin1p_log_sum')
            log_op_digests = reader.graph_op_digests(op_type='Log')
            self.assertLen(log_op_digests, 1)
            self.assertEqual(reader.graph_by_id(log_op_digests[0].graph_id).name, 'log_sum')
            sin_op_digests = reader.graph_op_digests(op_type='Sin')
            self.assertLen(sin_op_digests, 1)
            self.assertEqual(reader.graph_by_id(sin_op_digests[0].graph_id).name, 'sin1p_log_sum')
            for op_digest in add_op_digests + log_op_digests + sin_op_digests:
                self.assertLen(op_digest.output_tensor_ids, 1)
                self.assertGreaterEqual(op_digest.output_tensor_ids[0], 0)
                (_, stack_frames) = reader.read_graph_op_creation_stack_trace(op_digest)
                self._verifyStackFrames(stack_frames)
            graph_exec_traces = [trace for trace in reader.graph_execution_traces() if trace.op_type not in ['Const', 'Placeholder']]
            executed_op_types = [digest.op_type for digest in graph_exec_traces]
            self.assertEqual(executed_op_types, ['AddV2', 'Log', 'AddV2', 'Sin'])
            self.assertEqual(reader.graph_by_id(graph_exec_traces[0].graph_ids[-1]).name, 'log_sum')
            self.assertEqual(reader.graph_by_id(graph_exec_traces[0].graph_ids[-2]).name, 'sin1p_log_sum')
            self.assertEqual(reader.graph_by_id(graph_exec_traces[1].graph_ids[-1]).name, 'log_sum')
            self.assertEqual(reader.graph_by_id(graph_exec_traces[1].graph_ids[-2]).name, 'sin1p_log_sum')
            self.assertEqual(reader.graph_by_id(graph_exec_traces[2].graph_ids[-1]).name, 'sin1p_log_sum')
            self.assertEqual(reader.graph_by_id(graph_exec_traces[3].graph_ids[-1]).name, 'sin1p_log_sum')
            if tensor_debug_mode == 'NO_TENSOR':
                for trace in graph_exec_traces:
                    self.assertIsNone(trace.debug_tensor_value)
            elif tensor_debug_mode == 'CURT_HEALTH':
                self.assertAllClose(graph_exec_traces[0].debug_tensor_value, [add_op_digests[0].output_tensor_ids[0], 0.0])
                self.assertAllClose(graph_exec_traces[1].debug_tensor_value, [log_op_digests[0].output_tensor_ids[0], 0.0])
                self.assertAllClose(graph_exec_traces[2].debug_tensor_value, [add_op_digests[1].output_tensor_ids[0], 0.0])
                self.assertAllClose(graph_exec_traces[3].debug_tensor_value, [sin_op_digests[0].output_tensor_ids[0], 0.0])
            elif tensor_debug_mode == 'CONCISE_HEALTH':
                self.assertAllClose(graph_exec_traces[0].debug_tensor_value, [add_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
                self.assertAllClose(graph_exec_traces[1].debug_tensor_value, [log_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
                self.assertAllClose(graph_exec_traces[2].debug_tensor_value, [add_op_digests[1].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
                self.assertAllClose(graph_exec_traces[3].debug_tensor_value, [sin_op_digests[0].output_tensor_ids[0], 1.0, 0.0, 0.0, 0.0])
            elif tensor_debug_mode == 'FULL_HEALTH':
                self.assertAllClose(graph_exec_traces[0].debug_tensor_value, [add_op_digests[0].output_tensor_ids[0], -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
                self.assertAllClose(graph_exec_traces[1].debug_tensor_value, [log_op_digests[0].output_tensor_ids[0], -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
                self.assertAllClose(graph_exec_traces[2].debug_tensor_value, [add_op_digests[1].output_tensor_ids[0], -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
                self.assertAllClose(graph_exec_traces[3].debug_tensor_value, [sin_op_digests[0].output_tensor_ids[0], -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
            elif tensor_debug_mode == 'SHAPE':
                self.assertAllClose(graph_exec_traces[0].debug_tensor_value, [add_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
                self.assertAllClose(graph_exec_traces[1].debug_tensor_value, [log_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
                self.assertAllClose(graph_exec_traces[2].debug_tensor_value, [add_op_digests[1].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
                self.assertAllClose(graph_exec_traces[3].debug_tensor_value, [sin_op_digests[0].output_tensor_ids[0], 1, 0, 1, 0, 0, 0, 0, 0, 0])
            else:
                full_tensor_values = [reader.graph_execution_trace_to_tensor_value(trace) for trace in graph_exec_traces]
                self.assertAllClose(full_tensor_values[0], 5.0)
                self.assertAllClose(full_tensor_values[1], np.log(5.0))
                self.assertAllClose(full_tensor_values[2], np.log(5.0) + 1.0)
                self.assertAllClose(full_tensor_values[3], np.sin(np.log(5.0) + 1.0))

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('FullTensor', 'FULL_TENSOR'))
    @test_util.run_in_graph_and_eager_modes
    def testGraphOpConsumingRelationIsCaptured(self, tensor_debug_mode):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant([2.0, 2.0])
        y = constant_op.constant([3.0, 3.0])
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)

        @def_function.function
        def log_sum(x, y):
            if False:
                while True:
                    i = 10
            return math_ops.log(x + y)

        @def_function.function
        def maxindex_sin1p_log_sum(x, y):
            if False:
                for i in range(10):
                    print('nop')
            (_, indices) = array_ops.unique(math_ops.sin(1.0 + log_sum(x, y)))
            return math_ops.reduce_max(indices)
        maxindex = maxindex_sin1p_log_sum(x, y)
        self.assertAllEqual(maxindex, 0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            traces = reader.graph_execution_traces()
            add_traces = [trace for trace in traces if trace.op_type == 'AddV2']
            log_traces = [trace for trace in traces if trace.op_type == 'Log']
            sin_traces = [trace for trace in traces if trace.op_type == 'Sin']
            unique_traces = [trace for trace in traces if trace.op_type == 'Unique']
            max_traces = [trace for trace in traces if trace.op_type == 'Max']
            self.assertLen(add_traces, 2)
            self.assertLen(log_traces, 1)
            self.assertLen(sin_traces, 1)
            self.assertLen(unique_traces, 2)
            self.assertLen(max_traces, 1)
            graph = reader.graph_by_id(add_traces[0].graph_id)
            self.assertEqual(graph.get_op_consumers(add_traces[0].op_name), [(0, log_traces[0].op_name, 0)])
            graph = reader.graph_by_id(add_traces[1].graph_id)
            self.assertEqual(graph.get_op_consumers(add_traces[1].op_name), [(0, sin_traces[0].op_name, 0)])
            self.assertEqual(graph.get_op_consumers(sin_traces[0].op_name), [(0, unique_traces[0].op_name, 0)])
            self.assertEqual(graph.get_op_consumers(unique_traces[0].op_name), [(1, max_traces[0].op_name, 0)])

    def testCapturingExecutedGraphIdsOfTwoCompilationsOfSameFunction(self):
        if False:
            i = 10
            return i + 15
        'Test correct executed IDs of two FuncGraphs from the same Py function.'
        x_float32 = constant_op.constant(np.array(3.5, dtype=np.float32))
        x_float64 = constant_op.constant(np.array(4.5, dtype=np.float64))
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='NO_TENSOR')

        @def_function.function
        def ceil_times_two(x):
            if False:
                while True:
                    i = 10
            return math_ops.ceil(x) * 2.0
        self.assertAllClose(ceil_times_two(x_float32), 8.0)
        self.assertAllClose(ceil_times_two(x_float64), 10.0)
        self.assertAllClose(ceil_times_two(x_float32), 8.0)
        self.assertAllClose(ceil_times_two(x_float64), 10.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            executions = reader.executions()
            self.assertLen(executions, 4)
            for execution in executions:
                self.assertStartsWith(execution.op_type, '__inference_ceil_times_two_')
            executed_graph_ids = [execution.graph_id for execution in executions]
            self.assertEqual(executed_graph_ids[0], executed_graph_ids[2])
            self.assertEqual(executed_graph_ids[1], executed_graph_ids[3])
            self.assertNotEqual(executed_graph_ids[0], executed_graph_ids[1])
            self.assertNotEqual(executed_graph_ids[2], executed_graph_ids[3])
            for executed_graph_id in executed_graph_ids:
                self.assertEqual(reader.graph_by_id(executed_graph_id).name, 'ceil_times_two')

    def testCapturingExecutedGraphIdsOfDuplicateFunctionNames(self):
        if False:
            print('Hello World!')
        'Two FuncGraphs compiled from Python functions with identical names.'
        x = constant_op.constant(np.array(3.5, dtype=np.float32))
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='NO_TENSOR')

        class TestClass(object):

            @def_function.function
            def ceil_times_two(self, x):
                if False:
                    while True:
                        i = 10
                return math_ops.ceil(x) * 2.0
        test_object_1 = TestClass()
        test_object_2 = TestClass()
        self.assertAllClose(test_object_1.ceil_times_two(x), 8.0)
        self.assertAllClose(test_object_2.ceil_times_two(x), 8.0)
        self.assertAllClose(test_object_1.ceil_times_two(x), 8.0)
        self.assertAllClose(test_object_2.ceil_times_two(x), 8.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            executions = reader.executions()
            self.assertLen(executions, 4)
            for execution in executions:
                self.assertStartsWith(execution.op_type, '__inference_ceil_times_two_')
            executed_graph_ids = [execution.graph_id for execution in executions]
            self.assertEqual(executed_graph_ids[0], executed_graph_ids[2])
            self.assertEqual(executed_graph_ids[1], executed_graph_ids[3])
            self.assertNotEqual(executed_graph_ids[0], executed_graph_ids[1])
            self.assertNotEqual(executed_graph_ids[2], executed_graph_ids[3])
            for executed_graph_id in executed_graph_ids:
                self.assertEqual(reader.graph_by_id(executed_graph_id).name, 'ceil_times_two')

    @parameterized.named_parameters(('AddV2', 'AddV2'), ('Log', 'Log'), ('AddV2AndLog', '(AddV2|Log)'))
    @test_util.run_in_graph_and_eager_modes
    def testOpRegex(self, op_regex):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(2.0)
        y = constant_op.constant(3.0)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='FULL_TENSOR', op_regex=op_regex)

        @def_function.function
        def log_sum(x, y):
            if False:
                for i in range(10):
                    print('nop')
            return math_ops.log(x + y)

        @def_function.function
        def sin1p_log_sum(x, y):
            if False:
                return 10
            return math_ops.sin(1.0 + log_sum(x, y))
        self.assertAllClose(self.evaluate(sin1p_log_sum(x, y)), np.sin(1.0 + np.log(5.0)))
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            graph_op_digests = reader.graph_op_digests()
            op_types = [digest.op_type for digest in graph_op_digests]
            self.assertIn('AddV2', op_types)
            self.assertIn('Log', op_types)
            self.assertIn('Sin', op_types)
            graph_exec_digests = reader.graph_execution_traces(digest=True)
            executed_op_types = [digest.op_type for digest in graph_exec_digests]
            tensor_values = [reader.graph_execution_trace_to_tensor_value(digest) for digest in graph_exec_digests]
            if op_regex == 'AddV2':
                self.assertEqual(executed_op_types, ['AddV2', 'AddV2'])
                self.assertLen(tensor_values, 2)
                self.assertAllClose(tensor_values[0], 5.0)
                self.assertAllClose(tensor_values[1], np.log(5.0) + 1.0)
            elif op_regex == 'Log':
                self.assertEqual(executed_op_types, ['Log'])
                self.assertLen(tensor_values, 1)
                self.assertAllClose(tensor_values[0], np.log(5.0))
            else:
                self.assertEqual(executed_op_types, ['AddV2', 'Log', 'AddV2'])
                self.assertLen(tensor_values, 3)
                self.assertAllClose(tensor_values[0], 5.0)
                self.assertAllClose(tensor_values[1], np.log(5.0))
                self.assertAllClose(tensor_values[2], np.log(5.0) + 1.0)

    def testIncorrectTensorDTypeArgFormatLeadsToError(self):
        if False:
            print('Hello World!')
        with self.assertRaisesRegex(ValueError, '.*expected.*list.*tuple.*callable.*but received.*\\{\\}'):
            dumping_callback.enable_dump_debug_info(self.dump_root, tensor_dtypes=dict())
        with self.assertRaisesRegex(ValueError, '.*expected.*list.*tuple.*callable.*but received.*'):
            dumping_callback.enable_dump_debug_info(self.dump_root, tensor_dtypes='float32')
        with self.assertRaisesRegex(ValueError, '.*expected.*list.*tuple.*callable.*but received.*'):
            dumping_callback.enable_dump_debug_info(self.dump_root, tensor_dtypes=dtypes.float32)
        with self.assertRaises(TypeError):
            dumping_callback.enable_dump_debug_info(self.dump_root, tensor_dtypes=[lambda dtype: dtype.is_floating, lambda dtype: dtype.is_integer])

    @parameterized.named_parameters(('float', [dtypes.float32], None), ('float_only_sum', ['float32'], 'Sum'), ('float_no_sum', (dtypes.float32,), '(?!Sum)'), ('int', [dtypes.int32], None), ('int_via_lambda', lambda dtype: dtype.is_integer, None), ('exclude_Sum', None, '(?!Sum)'), ('All', None, None))
    @test_util.run_in_graph_and_eager_modes
    def testTensorDTypesAndOpRegexFilters(self, tensor_dtypes, op_regex):
        if False:
            for i in range(10):
                print('nop')
        xs = constant_op.constant([2.0, 6.0, 8.0, 1.0, 2.0], dtype=dtypes.float32)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='FULL_TENSOR', tensor_dtypes=tensor_dtypes, op_regex=op_regex)

        @def_function.function
        def unique_sum(xs):
            if False:
                print('Hello World!')
            'Sum over the unique values, for testing.'
            (unique_xs, indices) = array_ops.unique(xs)
            return (math_ops.reduce_sum(unique_xs), indices)
        (y, indices) = self.evaluate(unique_sum(xs))
        self.assertAllClose(y, 17.0)
        self.assertAllEqual(indices, [0, 1, 2, 3, 0])
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            graph_exec_digests = reader.graph_execution_traces(digest=True)
            executed_op_types = [digest.op_type for digest in graph_exec_digests if digest.op_type not in ('Const', 'Placeholder')]
            tensor_values = [reader.graph_execution_trace_to_tensor_value(digest) for digest in graph_exec_digests if digest.op_type not in ('Const', 'Placeholder')]
            if tensor_dtypes == [dtypes.float32] and (not op_regex):
                self.assertEqual(executed_op_types, ['Unique', 'Sum'])
                self.assertLen(tensor_values, 2)
                self.assertAllClose(tensor_values[0], [2, 6, 8, 1])
                self.assertAllClose(tensor_values[1], 17.0)
            elif tensor_dtypes == ['float32'] and op_regex == 'Sum':
                self.assertEqual(executed_op_types, ['Sum'])
                self.assertLen(tensor_values, 1)
                self.assertAllClose(tensor_values[0], 17.0)
            elif tensor_dtypes == (dtypes.float32,) and op_regex == '(?!Sum)':
                self.assertEqual(executed_op_types, ['Unique'])
                self.assertLen(tensor_values, 1)
                self.assertAllClose(tensor_values[0], [2, 6, 8, 1])
            elif tensor_dtypes == [dtypes.int32] and (not op_regex):
                self.assertEqual(executed_op_types, ['Unique'])
                self.assertLen(tensor_values, 1)
                self.assertAllEqual(tensor_values[0], [0, 1, 2, 3, 0])
            elif callable(tensor_dtypes) and (not op_regex):
                self.assertEqual(executed_op_types, ['Unique'])
                self.assertLen(tensor_values, 1)
                self.assertAllEqual(tensor_values[0], [0, 1, 2, 3, 0])
            elif not tensor_dtypes and op_regex == '(?!Sum)':
                self.assertEqual(executed_op_types, ['Unique', 'Unique'])
                self.assertLen(tensor_values, 2)
                self.assertAllClose(tensor_values[0], [2, 6, 8, 1])
                self.assertAllEqual(tensor_values[1], [0, 1, 2, 3, 0])
            else:
                self.assertEqual(executed_op_types, ['Unique', 'Unique', 'Sum'])
                self.assertLen(tensor_values, 3)
                self.assertAllClose(tensor_values[0], [2, 6, 8, 1])
                self.assertAllEqual(tensor_values[1], [0, 1, 2, 3, 0])
                self.assertAllClose(tensor_values[2], 17)

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('CurtHealth', 'CURT_HEALTH'), ('FullTensor', 'FULL_TENSOR'))
    @test_util.run_in_graph_and_eager_modes
    def testFunctionExecutionWithControlFlow(self, tensor_debug_mode):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(0.5, dtype=dtypes.float32)
        times = constant_op.constant(4, dtype=dtypes.int32)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)

        @def_function.function
        def iterative_doubling(x, times):
            if False:
                while True:
                    i = 10
            i = constant_op.constant(0, dtype=dtypes.int32)
            while i < times:
                x = x * 2.0
                i += 1
            return x
        self.assertAllClose(self.evaluate(iterative_doubling(x, times)), 8.0)
        writer.FlushNonExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            graph_op_digests = reader.graph_op_digests()
            op_types = [digest.op_type for digest in graph_op_digests]
            self.assertIn('Less', op_types)
            self.assertIn('Mul', op_types)
            self.assertIn('AddV2', op_types)
            self.assertEqual(reader.num_executions(), 0)
            self.assertEqual(reader.num_graph_execution_traces(), 0)
            writer.FlushExecutionFiles()
            reader.update()
            if context.executing_eagerly():
                executions = reader.executions()
                self.assertLen(executions, 1)
                executed_op_types = [execution.op_type for execution in executions]
                self.assertIn('iterative_doubling', executions[0].op_type)
                execution = executions[0]
                self.assertLen(execution.input_tensor_ids, 2)
                self.assertLen(execution.output_tensor_ids, 1)
                self.assertEqual(debug_event_pb2.TensorDebugMode.keys()[execution.tensor_debug_mode], tensor_debug_mode)
                if tensor_debug_mode == 'FULL_TENSOR':
                    tensor_values = reader.execution_to_tensor_values(execution)
                    self.assertAllClose(tensor_values, [8.0])
            graph_exec_traces = reader.graph_execution_traces()
            executed_op_types = [trace.op_type for trace in graph_exec_traces if trace.op_type != 'Const']
            if tensor_debug_mode != 'CURT_HEALTH':
                self.assertEqual(executed_op_types.count('Less'), 5)
                self.assertEqual(executed_op_types[-1], 'Less')
                self.assertIn('AddV2', executed_op_types)
                for trace in graph_exec_traces:
                    self.assertEqual(trace.output_slot, 0)
            self.assertEqual(executed_op_types.count('Mul'), 4)
            tensor_values = [reader.graph_execution_trace_to_tensor_value(trace) for trace in graph_exec_traces]
            if tensor_debug_mode == 'NO_TENSOR':
                for tensor_value in tensor_values:
                    self.assertAllEqual(tensor_value, [])
            elif tensor_debug_mode == 'CURT_HEALTH':
                for trace in graph_exec_traces:
                    tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                    self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0.0])
            elif tensor_debug_mode == 'FULL_TENSOR':
                less_values = [reader.graph_execution_trace_to_tensor_value(trace) for trace in graph_exec_traces if trace.op_type == 'Less']
                self.assertAllEqual(less_values, [True, True, True, True, False])
                mul_values = [reader.graph_execution_trace_to_tensor_value(trace) for trace in graph_exec_traces if trace.op_type == 'Mul']
                self.assertAllClose(mul_values, [1.0, 2.0, 4.0, 8.0])

    def testCallingEnableTracingTwiceWithTheSameDumpRootIsIdempotent(self):
        if False:
            while True:
                i = 10
        x = constant_op.constant([10.0, 12.0, 10.0])
        dumping_callback.enable_dump_debug_info(self.dump_root)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root)
        for _ in range(2):
            array_ops.unique(x)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            executions = reader.executions()
            self.assertLen(executions, 2)
            for execution in executions:
                self.assertGreater(execution.wall_time, 0)
                self.assertEqual(execution.op_type, 'Unique')
                self.assertEqual(execution.num_outputs, 2)
                (_, stack_frames) = reader.read_execution_stack_trace(execution)
                self._verifyStackFrames(stack_frames)

    def testCallingEnableTracingTwiceWithDifferentDumpRootsOverwrites(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant([10.0, 12.0, 10.0])
        dumping_callback.enable_dump_debug_info(self.dump_root)
        new_dump_root = self.dump_root + '_new_dump_root'
        writer = dumping_callback.enable_dump_debug_info(new_dump_root)
        for _ in range(2):
            array_ops.unique(x)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(new_dump_root) as reader:
            reader.update()
            executions = reader.executions()
            self.assertLen(executions, 2)
            for execution in executions:
                self.assertGreater(execution.wall_time, 0)
                self.assertEqual(execution.op_type, 'Unique')
                self.assertEqual(execution.num_outputs, 2)
                (_, stack_frames) = reader.read_execution_stack_trace(execution)
                self._verifyStackFrames(stack_frames)
        with debug_events_reader.DebugDataReader(self.dump_root) as old_dump_root_reader:
            old_dump_root_reader.update()
            self.assertEqual(old_dump_root_reader.num_executions(), 0)
            self.assertFalse(old_dump_root_reader.outermost_graphs())

    def testCallingEnableRepeatedlyWithDifferentTensorDebugMode(self):
        if False:
            print('Hello World!')
        'Assert calling enable_dump_debug_info() with two tensor-debug modes.\n\n    It should lead to overwriting of the previously-configured mode.\n    '
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='NO_TENSOR')

        @def_function.function
        def add_1_divide_by_2(x):
            if False:
                print('Hello World!')
            return (x + 1.0) / 2.0
        self.assertAllClose(add_1_divide_by_2(constant_op.constant(4.0)), 2.5)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            graph_exec_digests = reader.graph_execution_traces(digest=True)
            tensor_values = [reader.graph_execution_trace_to_tensor_value(digest) for digest in graph_exec_digests]
            for tensor_value in tensor_values:
                self.assertAllEqual(tensor_value, [])
        with self.assertRaisesRegex(ValueError, 'already.*NO_TENSOR.*FULL_TENSOR.*not be honored'):
            dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='FULL_TENSOR')

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('FullTensor', 'FULL_TENSOR'))
    def testDisableTracingWorks(self, tensor_debug_mode):
        if False:
            return 10
        x = constant_op.constant([10.0, 12.0, 10.0])
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)
        dumping_callback.disable_dump_debug_info()
        for _ in range(2):
            array_ops.unique(x)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            self.assertEqual(reader.num_executions(), 0)
            self.assertEqual(reader.num_graph_execution_traces(), 0)
            self.assertFalse(reader.outermost_graphs())

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('CurtHealth', 'CURT_HEALTH'), ('ConciseHealth', 'CONCISE_HEALTH'), ('FullHealth', 'FULL_HEALTH'), ('Shape', 'SHAPE'), ('FullTensor', 'FULL_TENSOR'))
    def testMultiThreadedExecutionWithSameSetting(self, tensor_debug_mode):
        if False:
            return 10
        'Dumping from multiple threads using the same setting.'
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)
        x = variables.Variable(10.0, dtype=dtypes.float32)
        y = variables.Variable(3.0, dtype=dtypes.float32)

        @def_function.function
        def increase_x():
            if False:
                print('Hello World!')
            return x.assign_add(y * 2.0)
        increase_x()
        num_threads = 3
        threads = []
        for _ in range(num_threads):
            threads.append(threading.Thread(target=increase_x))
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        self.assertAllClose(x.read_value(), 34.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            exec_digests = reader.executions(digest=True)
            prev_wall_time = 1
            for exec_digest in exec_digests:
                self.assertGreaterEqual(exec_digest.wall_time, prev_wall_time)
                prev_wall_time = exec_digest.wall_time
            graph_exec_traces = reader.graph_execution_traces()
            executed_op_types = [trace.op_type for trace in graph_exec_traces]
            self.assertEqual(executed_op_types.count('Mul'), 1 + num_threads)
            self.assertEqual(executed_op_types.count('ReadVariableOp'), 2 * (1 + num_threads))
            for trace in graph_exec_traces:
                self.assertEqual(trace.output_slot, 0)
        tensor_values = [reader.graph_execution_trace_to_tensor_value(trace) for trace in graph_exec_traces]
        if tensor_debug_mode == 'NO_TENSOR':
            for tensor_value in tensor_values:
                self.assertAllEqual(tensor_value, [])
        elif tensor_debug_mode == 'CURT_HEALTH':
            for trace in graph_exec_traces:
                tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                self.assertAllClose(trace.debug_tensor_value, [tensor_id, 0])
        elif tensor_debug_mode == 'CONCISE_HEALTH':
            for trace in graph_exec_traces:
                tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                self.assertAllClose(trace.debug_tensor_value, [tensor_id, 1, 0, 0, 0])
        elif tensor_debug_mode == 'FULL_HEALTH':
            for trace in graph_exec_traces:
                tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                self.assertAllClose(trace.debug_tensor_value, [tensor_id, -1, 1, 0, 1, 0, 0, 0, 0, 0, 1])
        elif tensor_debug_mode == 'SHAPE':
            for trace in graph_exec_traces:
                if trace.op_type == 'Mul':
                    tensor_id = reader.graph_execution_trace_to_tensor_id(trace)
                    mul_value = reader.graph_execution_trace_to_tensor_value(trace)
                    self.assertAllClose(mul_value, [tensor_id, 1, 0, 1, 0, 0, 0, 0, 0, 0])
        elif tensor_debug_mode == 'FULL_TENSOR':
            mul_values = [reader.graph_execution_trace_to_tensor_value(trace) for trace in graph_exec_traces if trace.op_type == 'Mul']
            self.assertAllClose(mul_values, [6.0, 6.0, 6.0, 6.0])

    def testMultiThreadedDumpingWithDifferentSettings(self):
        if False:
            for i in range(10):
                print('nop')
        gpu_name = test_util.gpu_device_name()
        if gpu_name:
            self.skipTest('b/153671240: test is flaky on GPUs')
        dump_root_1 = os.path.join(self.dump_root, 'dump_root_1')
        dump_root_2 = os.path.join(self.dump_root, 'dump_root_2')
        v1 = variables.Variable(10.0, dtype=dtypes.float32)
        v2 = variables.Variable(3.0, dtype=dtypes.float32)

        def add_negative_v1_squared_to_itself():
            if False:
                print('Hello World!')
            writer = dumping_callback.enable_dump_debug_info(dump_root_1, tensor_debug_mode='FULL_TENSOR')
            for _ in range(3):
                v1.assign_add(-v1 ** 2.0)
            writer.FlushNonExecutionFiles()
            writer.FlushExecutionFiles()

        def add_negative_v2_squared_to_itself():
            if False:
                for i in range(10):
                    print('nop')
            writer = dumping_callback.enable_dump_debug_info(dump_root_2, tensor_debug_mode='FULL_TENSOR')
            v2_squared = v2 ** 2.0
            dumping_callback.disable_dump_debug_info()
            negative_v2_squared = -v2_squared
            v2.assign_add(negative_v2_squared)
            writer.FlushNonExecutionFiles()
            writer.FlushExecutionFiles()
        sub_thread = threading.Thread(target=add_negative_v2_squared_to_itself)
        sub_thread.start()
        add_negative_v1_squared_to_itself()
        sub_thread.join()
        self.assertAllClose(v1.read_value(), -67084290.0)
        self.assertAllClose(v2.read_value(), -6.0)
        with debug_events_reader.DebugDataReader(dump_root_1) as reader:
            reader.update()
            exec_digests = reader.executions(digest=True)
            v1_squared_values = [reader.execution_to_tensor_values(digest) for digest in exec_digests if digest.op_type == 'Pow']
            negative_v1_squared_values = [reader.execution_to_tensor_values(digest) for digest in exec_digests if digest.op_type == 'Neg']
            self.assertAllClose(v1_squared_values, [[100.0], [8100.0], [67076100.0]])
            self.assertAllClose(negative_v1_squared_values, [[-100.0], [-8100.0], [-67076100.0]])
        with debug_events_reader.DebugDataReader(dump_root_2) as reader:
            reader.update()
            exec_digests = reader.executions(digest=True)
            executed_op_types = [digest.op_type for digest in exec_digests]
            self.assertNotIn('Neg', executed_op_types)
            v2_squared_values = [reader.execution_to_tensor_values(digest) for digest in exec_digests if digest.op_type == 'Pow']
            self.assertAllClose(v2_squared_values, [[9.0]])

    @test_util.run_in_graph_and_eager_modes
    def testNestedContextIsCapturedByGraphOpCreationHistory(self):
        if False:
            for i in range(10):
                print('nop')
        x = constant_op.constant(2.0, dtype=dtypes.float32)
        times = constant_op.constant(4, dtype=dtypes.int32)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode='NO_TENSOR')

        @def_function.function
        def iterative_doubling(x, times):
            if False:
                return 10
            i = constant_op.constant(0, dtype=dtypes.int32)
            while i < times:
                x = x * 2.0 - 1.0
                i += 1
            return x
        self.assertAllClose(self.evaluate(iterative_doubling(x, times)), 17.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            less_op_digest = reader.graph_op_digests(op_type='Less')[-1]
            mul_op_digest = reader.graph_op_digests(op_type='Mul')[-1]
            sub_op_digest = reader.graph_op_digests(op_type='Sub')[-1]
            self.assertNotEqual(less_op_digest.graph_id, mul_op_digest.graph_id)
            self.assertNotEqual(less_op_digest.graph_id, sub_op_digest.graph_id)
            self.assertEqual(mul_op_digest.graph_id, sub_op_digest.graph_id)

    @parameterized.named_parameters(('NoTensor', 'NO_TENSOR'), ('Shape', 'SHAPE'), ('FullTensor', 'FULL_TENSOR'))
    @test_util.run_in_graph_and_eager_modes
    def testGraphInputTracingWorksWithConstAndPlaceholderTensors(self, tensor_debug_mode):
        if False:
            i = 10
            return i + 15
        x = constant_op.constant(2.0)
        writer = dumping_callback.enable_dump_debug_info(self.dump_root, tensor_debug_mode=tensor_debug_mode)

        @def_function.function
        def func(x):
            if False:
                print('Hello World!')
            return (x + constant_op.constant(4.0)) / x
        self.assertAllClose(self.evaluate(func(x)), 3.0)
        writer.FlushNonExecutionFiles()
        writer.FlushExecutionFiles()
        with debug_events_reader.DebugDataReader(self.dump_root) as reader:
            reader.update()
            graph_op_digests = reader.graph_op_digests()
            placeholder_op_name = None
            const_op_name = None
            add_op_name = None
            div_op_name = None
            for op_digest in graph_op_digests:
                if op_digest.op_type == 'Placeholder':
                    placeholder_op_name = op_digest.op_name
                elif op_digest.op_type == 'Const':
                    const_op_name = op_digest.op_name
                elif op_digest.op_type == 'AddV2':
                    add_op_name = op_digest.op_name
                    self.assertLen(op_digest.input_names, 2)
                    self.assertEqual(op_digest.input_names[0], placeholder_op_name + ':0')
                    self.assertEqual(op_digest.input_names[1], const_op_name + ':0')
                elif op_digest.op_type == 'RealDiv':
                    div_op_name = op_digest
                    self.assertLen(op_digest.input_names, 2)
                    self.assertEqual(op_digest.input_names[0], add_op_name + ':0')
                    self.assertEqual(op_digest.input_names[1], placeholder_op_name + ':0')
            self.assertTrue(add_op_name)
            self.assertTrue(div_op_name)
if __name__ == '__main__':
    ops.enable_eager_execution()
    googletest.main()