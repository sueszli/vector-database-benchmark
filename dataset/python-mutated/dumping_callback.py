"""Dumping op callbacks: Enables dump-based features in tfdbg v2."""
import atexit
import os
import re
import socket
import threading
import uuid
from tensorflow.core.framework import graph_debug_info_pb2
from tensorflow.core.framework import tensor_pb2
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import debug_events_writer
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import function as function_lib
from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util import tf_stack
from tensorflow.python.util.tf_export import tf_export
_state = threading.local()
DEFAULT_TENSOR_DEBUG_MODE = 'NO_TENSOR'
_FUNCTION_PREFIXES = (compat.as_bytes(function_lib._FORWARD_PREFIX), compat.as_bytes(function_lib._BACKWARD_PREFIX), compat.as_bytes(function_lib._INFERENCE_PREFIX))

def is_op_type_function(op_type):
    if False:
        return 10
    return compat.as_bytes(op_type).startswith(_FUNCTION_PREFIXES)

@ops.RegisterGradient('DebugIdentityV2')
def _debug_identity_v2_grad(op, dy):
    if False:
        return 10
    'Gradient function for the DebugIdentityV2 op.'
    del op
    return dy

def _get_tfdbg_run_id():
    if False:
        while True:
            i = 10
    return str(uuid.uuid4())[:8]

def _get_id():
    if False:
        i = 10
        return i + 15
    'Get a short unique ID.'
    return str(uuid.uuid4())

def _concrete_tensor_to_proto(tensor):
    if False:
        while True:
            i = 10
    return tensor_util.make_tensor_proto(tensor.numpy())

class _DumpingCallback(object):
    """An object holding the states surrounding the dumping callback."""

    def __init__(self, dump_root, tensor_debug_mode, circular_buffer_size, op_regex, tensor_dtypes):
        if False:
            return 10
        self._dump_root = dump_root
        self._tfdbg_run_id = _get_tfdbg_run_id()
        self._tensor_debug_mode = tensor_debug_mode
        self._circular_buffer_size = circular_buffer_size
        self._op_regex = op_regex
        self._tensor_dtypes = tensor_dtypes
        self._hostname = socket.gethostname()
        self._source_file_paths = []
        self._stack_frame_to_id = dict()
        self._context_to_id = dict()
        self._function_to_graph_id = dict()
        self._op_type_to_context_id = dict()
        self._symbolic_tensor_counter = 0
        self._tensor_aliases = dict()
        self._source_file_paths_lock = threading.Lock()
        self._stack_frame_to_id_lock = threading.Lock()
        self._context_lock = threading.Lock()
        self._symbolic_tensor_counter_lock = threading.Lock()
        self._placeholder_to_debug_tensor = object_identity.ObjectIdentityDictionary()
        self._writer = None

    def function_callback(self, function):
        if False:
            while True:
                i = 10
        'A callback to be called on creation of ConcreteFunctions.'
        graph_id = self._get_context_id(function.graph)
        with self._context_lock:
            self._function_to_graph_id[function] = graph_id

    @property
    def dump_root(self):
        if False:
            print('Hello World!')
        return self._dump_root

    @dump_root.setter
    def dump_root(self, dump_root):
        if False:
            for i in range(10):
                print('nop')
        if self._dump_root != dump_root:
            self._dump_root = dump_root
            self._writer = None

    @property
    def tfdbg_run_id(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tfdbg_run_id

    @property
    def tensor_debug_mode(self):
        if False:
            for i in range(10):
                print('nop')
        return self._tensor_debug_mode

    @property
    def circular_buffer_size(self):
        if False:
            for i in range(10):
                print('nop')
        return self._circular_buffer_size

    def get_writer(self):
        if False:
            i = 10
            return i + 15
        'Get the debug events writer for the currently configured dump root.'
        if not self._writer:
            self._writer = debug_events_writer.DebugEventsWriter(self._dump_root, self._tfdbg_run_id, circular_buffer_size=self._circular_buffer_size)
        return self._writer

    def _get_context_id(self, context):
        if False:
            return 10
        'Get a unique ID for an op-construction context (e.g., a graph).\n\n    If the graph has been encountered before, reuse the same unique ID.\n    When encountering a new context (graph), this methods writes a DebugEvent\n    proto with the debugged_graph field to the proper DebugEvent file.\n\n    Args:\n      context: A context to get the unique ID for. Must be hashable. E.g., a\n        Graph object.\n\n    Returns:\n      A unique ID for the context.\n    '
        if context in self._context_to_id:
            return self._context_to_id[context]
        graph_is_new = False
        with self._context_lock:
            if context not in self._context_to_id:
                graph_is_new = True
                context_id = _get_id()
                self._context_to_id[context] = context_id
        if graph_is_new:
            self.get_writer().WriteDebuggedGraph(debug_event_pb2.DebuggedGraph(graph_id=context_id, graph_name=getattr(context, 'name', None), outer_context_id=self._get_outer_context_id(context)))
        return self._context_to_id[context]

    def _get_outer_context_id(self, graph):
        if False:
            for i in range(10):
                print('nop')
        'Get the ID of the immediate outer context of the input graph.\n\n    Args:\n      graph: The graph (context) in question.\n\n    Returns:\n      If an outer context exists, the immediate outer context name as a string.\n      If such as outer context does not exist (i.e., `graph` is itself\n      outermost), `None`.\n    '
        if hasattr(graph, 'outer_graph') and graph.outer_graph:
            return self._get_context_id(graph.outer_graph)
        else:
            return None

    def _write_source_file_content(self, file_path):
        if False:
            i = 10
            return i + 15
        'Send the content of a source file via debug-events writer.\n\n    Args:\n      file_path: Path to the source file.\n\n    Returns:\n      An int index for the file.\n    '
        if file_path in self._source_file_paths:
            return self._source_file_paths.index(file_path)
        with self._source_file_paths_lock:
            if file_path not in self._source_file_paths:
                lines = None
                if source_utils.is_extension_uncompiled_python_source(file_path):
                    try:
                        (lines, _) = source_utils.load_source(file_path)
                    except IOError as e:
                        logging.warn('Failed to read source code from path: %s. Reason: %s', file_path, e)
                writer = self.get_writer()
                writer.WriteSourceFile(debug_event_pb2.SourceFile(file_path=file_path, host_name=self._hostname, lines=lines))
                self._source_file_paths.append(file_path)
            return self._source_file_paths.index(file_path)

    def _process_stack_frames(self):
        if False:
            while True:
                i = 10
        'Process stack frames.\n\n    Send the content of source-files, on a best-effort basis.\n\n    Returns:\n      A list of stack frame IDs.\n    '
        stack_frames = tf_stack.extract_stack()
        stack_frame_ids = []
        writer = None
        for (file_path, lineno, func, _) in stack_frames:
            abs_path = os.path.abspath(file_path)
            if (abs_path, lineno, func) in self._stack_frame_to_id:
                stack_frame_ids.append(self._stack_frame_to_id[abs_path, lineno, func])
                continue
            with self._stack_frame_to_id_lock:
                if (abs_path, lineno, func) not in self._stack_frame_to_id:
                    stack_frame_id = _get_id()
                    self._stack_frame_to_id[abs_path, lineno, func] = stack_frame_id
                    file_index = self._write_source_file_content(abs_path)
                    file_line_col = graph_debug_info_pb2.GraphDebugInfo.FileLineCol(file_index=file_index, line=lineno, func=func)
                    stack_frame_with_id = debug_event_pb2.StackFrameWithId(id=stack_frame_id, file_line_col=file_line_col)
                    writer = self.get_writer()
                    writer.WriteStackFrameWithId(stack_frame_with_id)
                stack_frame_ids.append(self._stack_frame_to_id[abs_path, lineno, func])
        code_location = debug_event_pb2.CodeLocation(host_name=self._hostname, stack_frame_ids=stack_frame_ids)
        return code_location

    def _process_v1_graph_mode_tensor(self, op_type, tensor, debug_tensor, tensor_debug_mode):
        if False:
            for i in range(10):
                print('nop')
        'For V1 graph mode, determine what tensor to output from callback.\n\n    Args:\n      op_type: Type of the op that outputs the original symbolic tensor.\n      tensor: The original output symbolic tensor.\n      debug_tensor: The debugger-instrumented tensor.\n      tensor_debug_mode: Debug mode used, a tfdbg TensorDebugMode enum.\n\n    Returns:\n      A symbolic tensor to be returned by the dumping op_callback.\n    '
        if op_type in ('Placeholder', 'PlaceholderWithDefault'):
            self._placeholder_to_debug_tensor[tensor] = debug_tensor
            return tensor
        elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR and op_type != 'Const':
            self._tensor_aliases[debug_tensor.name] = tensor.name
            return debug_tensor
        else:
            with self._symbolic_tensor_counter_lock:
                identity_name = 'tfdbg_identity_%d' % self._symbolic_tensor_counter
            identity = array_ops.identity(tensor, name=identity_name)
            identity.op._add_control_input(debug_tensor.op)
            self._tensor_aliases[identity.name] = tensor.name
            return identity

    def _instrument_symbolic_tensors(self, tensors, op_type, op_name, tfdbg_context_id, tensor_ids):
        if False:
            for i in range(10):
                print('nop')
        'Add debugging instrumentation for symbolic (i.e., non-eager) tensors.\n\n    The detailed fashion in which the tensors are instrumented is determined\n    by the tensor_debug_mode configured for the currently enabled dumping\n    callback.\n\n    Args:\n      tensors: A tuple of Tensors to instrument. It is assumed that their\n        ordering corresponds to the ordering of output tensors of an original\n        op. Output slot indices (0-based) will be generated based on the\n        ordering.\n      op_type: Type name of the op that emits the Tensors (e.g., "MatMul").\n      op_name: Name of the op that emits the Tensors (e.g., "dense_1/MatMul").\n      tfdbg_context_id: A unique ID for the context that the op belongs to\n        (e.g., a graph).\n      tensor_ids: A list of unique ID numbers for the tensors, for tfdbg\'s\n        internal use.\n\n    Returns:\n      Non-eager Tensors that override the `tensors` as the output of the op\n      that originally generated `tensors`. In some cases (e.g., non-V1 graph\n      mode), this may be `None`, as the instrumentation can simply rely on\n      automatic control dependencies (see `auto_control_deps.py`) instead of\n      tensor overriding.\n    '
        tensor_debug_mode = self._tensor_debug_mode
        debug_urls = ['file://%s' % self._dump_root]
        is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
        instrumented_tensors = [] if is_v1_graph_mode else None
        for (output_slot, tensor) in enumerate(tensors):
            with self._symbolic_tensor_counter_lock:
                debug_identity_name = 'DebugIdentityV2_%d' % self._symbolic_tensor_counter
            debug_identity_op_kwargs = {'tfdbg_context_id': tfdbg_context_id, 'op_name': op_name, 'output_slot': output_slot, 'tensor_debug_mode': self._tensor_debug_mode, 'debug_urls': debug_urls, 'name': debug_identity_name, 'circular_buffer_size': self._circular_buffer_size, 'tfdbg_run_id': self._tfdbg_run_id}
            if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
                if not self._should_dump_tensor(op_type, tensor.dtype) or not tensor.dtype.is_numpy_compatible:
                    if is_v1_graph_mode:
                        instrumented_tensors.append(tensor)
                    continue
                if is_v1_graph_mode and (not tensor.dtype.is_numpy_compatible):
                    instrumented_tensors.append(tensor)
                    continue
                debug_tensor = gen_debug_ops.debug_identity_v2(constant_op.constant([], dtype=dtypes.float32), **debug_identity_op_kwargs)
                if is_v1_graph_mode:
                    instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
            elif tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH, debug_event_pb2.TensorDebugMode.SHAPE):
                dtype = tensor.dtype
                dtype_is_dumpable = tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH) and dtype.is_floating or (tensor_debug_mode == debug_event_pb2.TensorDebugMode.SHAPE and (dtype.is_floating or dtype.is_integer or dtype.is_bool))
                if not self._should_dump_tensor(op_type, tensor.dtype) or not dtype_is_dumpable:
                    if is_v1_graph_mode:
                        instrumented_tensors.append(tensor)
                    continue
                debug_tensor = gen_debug_ops.debug_identity_v2(gen_debug_ops.debug_numeric_summary_v2(tensor, tensor_id=tensor_ids[output_slot], tensor_debug_mode=self._tensor_debug_mode, output_dtype=dtypes.float64), **debug_identity_op_kwargs)
                if is_v1_graph_mode:
                    instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
            elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
                if not self._should_dump_tensor(op_type, tensor.dtype) or not tensor.dtype.is_numpy_compatible:
                    if is_v1_graph_mode:
                        instrumented_tensors.append(tensor)
                    continue
                debug_tensor = gen_debug_ops.debug_identity_v2(tensor, **debug_identity_op_kwargs)
                if is_v1_graph_mode:
                    instrumented_tensors.append(self._process_v1_graph_mode_tensor(op_type, tensor, debug_tensor, tensor_debug_mode))
            else:
                raise NotImplementedError('Symbolic tensor instrumentation is not implemented for debug mode %s' % self._tensor_debug_mode)
        return instrumented_tensors

    def _dump_eager_tensors(self, tensors, op_type, input_tensor_ids, output_tensor_device_ids, graph_id=None):
        if False:
            while True:
                i = 10
        'Dump the value of eager tensors.\n\n    The destination of the dumping is determined by the dump_root of the\n    currently enabled dumping callback. The tensors may be transformed prior to\n    dumping (e.g., reduced as summary statistics such as minimum, maximum and\n    arithmetic  mean). The details of this transformation (if any) depends on\n    the tensor_debug_mode of the currently enabled dumping callback.\n\n    Args:\n      tensors: The EagerTensors whose values are to be dumped, with or without\n        value transform.\n      op_type: Type of the op that generates the tensors, as a string.\n      input_tensor_ids: IDs of the input EagerTensors to the op.\n      output_tensor_device_ids: Debugged-generated IDs for the devices on which\n        the output tensors are allocated, as a `list` of `int`s. Must match\n        `tensors` in length.\n      graph_id: ID of the executed graph, applicable only to eager execution of\n        a FuncGraph.\n\n    Returns:\n      A tfdbg Execution protocol buffer.\n    '
        tensor_debug_mode = self._tensor_debug_mode
        output_tensor_ids = [t._id for t in tensors]
        assert len(tensors) == len(output_tensor_device_ids)
        if tensor_debug_mode == debug_event_pb2.TensorDebugMode.NO_TENSOR:
            return debug_event_pb2.Execution(op_type=op_type, graph_id=graph_id, num_outputs=len(tensors), input_tensor_ids=input_tensor_ids, output_tensor_ids=output_tensor_ids, output_tensor_device_ids=output_tensor_device_ids, tensor_debug_mode=tensor_debug_mode, code_location=self._process_stack_frames())
        elif tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH, debug_event_pb2.TensorDebugMode.SHAPE, debug_event_pb2.TensorDebugMode.FULL_TENSOR):
            execution_proto = debug_event_pb2.Execution(op_type=op_type, num_outputs=len(tensors), graph_id=graph_id, input_tensor_ids=input_tensor_ids, output_tensor_ids=output_tensor_ids, output_tensor_device_ids=output_tensor_device_ids, tensor_debug_mode=tensor_debug_mode, code_location=self._process_stack_frames())
            for tensor in tensors:
                if self._should_dump_tensor(op_type, tensor.dtype) and tensor.dtype.is_numpy_compatible:
                    if tensor_debug_mode in (debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH):
                        if tensor.dtype.is_floating:
                            tensor_proto = _concrete_tensor_to_proto(gen_debug_ops.debug_numeric_summary_v2(tensor, tensor_debug_mode=tensor_debug_mode, output_dtype=dtypes.float64))
                        else:
                            tensor_proto = tensor_pb2.TensorProto()
                    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.SHAPE:
                        if tensor.dtype.is_floating or tensor.dtype.is_integer or tensor.dtype.is_bool:
                            tensor_proto = _concrete_tensor_to_proto(gen_debug_ops.debug_numeric_summary_v2(tensor, tensor_debug_mode=tensor_debug_mode, output_dtype=dtypes.float64))
                        else:
                            tensor_proto = tensor_pb2.TensorProto()
                    elif tensor_debug_mode == debug_event_pb2.TensorDebugMode.FULL_TENSOR:
                        tensor_proto = _concrete_tensor_to_proto(tensor)
                    if tensor_proto:
                        execution_proto.tensor_protos.append(tensor_proto)
            return execution_proto
        else:
            raise NotImplementedError('Tensor instrumentation is not implemented for debug mode %s yet ' % self._tensor_debug_mode)

    def callback(self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
        if False:
            return 10
        "Op callback for tracing (dumping) a TF program's execution."
        del attrs
        writer = self.get_writer()
        if graph:
            is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
            context_id = self._get_context_id(graph)
            output_tensor_ids = self._get_symbolic_tensor_ids(len(outputs))
            if op_type in ('Const', 'Placeholder', 'PlaceholderWithDefault'):
                op_name = outputs[0].name.split(':')[0]
            if is_v1_graph_mode:
                for input_tensor in inputs:
                    if input_tensor in self._placeholder_to_debug_tensor and outputs:
                        outputs[0].op._add_control_input(self._placeholder_to_debug_tensor[input_tensor].op)
            graph_op_creation = debug_event_pb2.GraphOpCreation(op_type=op_type, op_name=op_name, graph_name=graph.name if hasattr(graph, 'name') else None, graph_id=context_id, input_names=[self._lookup_tensor_name(input_tensor) for input_tensor in inputs], num_outputs=len(outputs), output_tensor_ids=output_tensor_ids, code_location=self._process_stack_frames())
            writer.WriteGraphOpCreation(graph_op_creation)
            if outputs and compat.as_bytes(op_type) not in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
                return self._instrument_symbolic_tensors(outputs, op_type, op_name, context_id, output_tensor_ids)
        else:
            op_type_bytes = compat.as_bytes(op_type)
            if op_type_bytes == b'DebugNumericSummaryV2':
                return None
            if op_type_bytes in op_callbacks_common.OP_CALLBACK_SKIP_OPS:
                return None
            context_id = self._func_graph_id_from_func_name(op_type)
            input_ids = [t._id for t in inputs]
            output_tensor_device_ids = [writer.RegisterDeviceAndGetId(output.device) for output in outputs] if outputs else []
            writer.WriteExecution(self._dump_eager_tensors(outputs, op_type, input_ids, output_tensor_device_ids, graph_id=context_id))

    def _lookup_tensor_name(self, tensor):
        if False:
            for i in range(10):
                print('nop')
        'Look up the name of a graph tensor.\n\n    This method maps the name of a debugger-generated Identity or\n    DebugIdentityV2 tensor to the name of the original instrumented tensor,\n    if `tensor` is such a debugger-created tensor.\n    Otherwise, it returns the name of `tensor` as is.\n\n    Args:\n      tensor: The graph tensor to look up the name for.\n\n    Returns:\n      Name of the original instrumented tensor as known to the debugger.\n    '
        return self._tensor_aliases.get(tensor.name, tensor.name)

    def _func_graph_id_from_func_name(self, op_type):
        if False:
            for i in range(10):
                print('nop')
        'Attempt to get the ID of a FuncGraph based on an op type name.\n\n    Also caches the ID for faster access later.\n\n    Args:\n      op_type: Op type string, which may be the name of a function.\n\n    Returns:\n      If the op_type name does not fit the pattern of a function name (e.g.,\n      one that starts with "__inference_"), `None` is returned immediately.\n      Else, if the FuncGraph is found, ID of the underlying FuncGraph is\n      returned as a string.\n      Else, `None` is returned.\n    '
        op_type = compat.as_bytes(op_type)
        if is_op_type_function(op_type):
            if op_type in self._op_type_to_context_id:
                return self._op_type_to_context_id[op_type]
            with self._context_lock:
                for function in self._function_to_graph_id:
                    if function.name == op_type:
                        graph_id = self._function_to_graph_id[function]
                        self._op_type_to_context_id[op_type] = graph_id
                        return graph_id
            return None
        else:
            return None

    def _get_symbolic_tensor_ids(self, num_tensors):
        if False:
            return 10
        tensor_ids = []
        if num_tensors:
            with self._symbolic_tensor_counter_lock:
                for _ in range(num_tensors):
                    self._symbolic_tensor_counter += 1
                    tensor_ids.append(self._symbolic_tensor_counter)
        return tensor_ids

    def _should_dump_tensor(self, op_type, dtype):
        if False:
            i = 10
            return i + 15
        'Determine if the given tensor\'s value will be dumped.\n\n    The determination is made given the configurations such as `op_regex`,\n    `tensor_dtypes`.\n\n    Args:\n      op_type: Name of the op\'s type, as a string (e.g., "MatMul").\n      dtype: The dtype of the tensor, as a `dtypes.DType` object.\n\n    Returns:\n      A bool indicating whether the tensor\'s value will be dumped.\n    '
        should_dump = True
        if self._op_regex:
            should_dump = should_dump and re.match(self._op_regex, op_type)
        if self._tensor_dtypes:
            if isinstance(self._tensor_dtypes, (list, tuple)):
                should_dump = should_dump and any((dtype == dtype_item for dtype_item in self._tensor_dtypes))
            else:
                should_dump = should_dump and self._tensor_dtypes(dtype)
        return should_dump

@tf_export('debugging.experimental.enable_dump_debug_info')
def enable_dump_debug_info(dump_root, tensor_debug_mode=DEFAULT_TENSOR_DEBUG_MODE, circular_buffer_size=1000, op_regex=None, tensor_dtypes=None):
    if False:
        return 10
    'Enable dumping debugging information from a TensorFlow program.\n\n  The debugging information is dumped to a directory on the file system\n  specified as `dump_root`.\n\n  The dumped debugging information can be ingested by debugger UIs.\n\n  The files in the dump directory contain the following information:\n    - TensorFlow Function construction (e.g., compilation of Python functions\n      decorated with @tf.function), the op types, names (if available), context,\n      the input and output tensors, and the associated stack traces.\n    - Execution of TensorFlow operations (ops) and Functions and their stack\n      traces, op types, names (if available) and contexts. In addition,\n      depending on the value of the `tensor_debug_mode` argument (see Args\n      section below), the value(s) of the output tensors or more concise\n      summaries of the tensor values will be dumped.\n    - A snapshot of Python source files involved in the execution of the\n      TensorFlow program.\n\n  Once enabled, the dumping can be disabled with the corresponding\n  `disable_dump_debug_info()` method under the same Python namespace.\n  Calling this method more than once with the same `dump_root` is idempotent.\n  Calling this method more than once with different `tensor_debug_mode`s\n  leads to a `ValueError`.\n  Calling this method more than once with different `circular_buffer_size`s\n  leads to a `ValueError`.\n  Calling this method with a different `dump_root` abolishes the\n  previously-enabled `dump_root`.\n\n  Usage example:\n\n  ```py\n  tf.debugging.experimental.enable_dump_debug_info(\'/tmp/my-tfdbg-dumps\')\n\n  # Code to build, train and run your TensorFlow model...\n  ```\n\n  NOTE: If your code is running on TPUs, be sure to call\n  `tf.config.set_soft_device_placement(True)` before calling\n  `tf.debugging.experimental.enable_dump_debug_info()` as this API uses\n  automatic outside compilation on TPUs. For example:\n\n  ```py\n  tf.config.set_soft_device_placement(True)\n  tf.debugging.experimental.enable_dump_debug_info(\n      logdir, tensor_debug_mode="FULL_HEALTH")\n\n  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=\'\')\n  strategy = tf.distribute.TPUStrategy(resolver)\n  with strategy.scope():\n    # ...\n  ```\n\n  Args:\n    dump_root: The directory path where the dumping information will be written.\n    tensor_debug_mode: Debug mode for tensor values, as a string.\n      The currently supported options are:\n      - "NO_TENSOR": (Default) Only traces the output tensors of all executed\n        ops (including those executed eagerly at the Python level or as a part\n        of a TensorFlow graph) and functions, while not extracting any\n        information from the values of the tensors.\n      - "CURT_HEALTH": For each floating-dtype tensor (e.g., tensors of dtypes\n        such as `float32`, `float64` and `bfloat16`), extracts a binary bit\n        indicating whether it contains any -infinity, +infinity or NaN.\n      - "CONCISE_HEALTH": For each floating-dtype tensor, extract total\n        element count, and counts of -infinity, +infinity and NaN elements.\n      - "FULL_HEALTH": For each floating-dtype tensor, extracts the dtype,\n        rank (number of dimensions), total element count, and counts of\n        -infinity, +infinity and NaN elements.\n      - "SHAPE": For each tensor (regardless of dtype), extracts its dtype,\n        rank, total element count and shape.\n    circular_buffer_size: Size of the circular buffers for execution events.\n      These circular buffers are designed to reduce the overhead of debugging\n      dumping. They hold the most recent debug events concerning eager execution\n      of ops and `tf.function`s and traces of tensor values computed inside\n      `tf.function`s. They are written to the file system only when the proper\n      flushing method is called (see description of return values below).\n      Expected to be an integer. If <= 0, the circular-buffer behavior will be\n      disabled, i.e., the execution debug events will be written to the file\n      writers in the same way as non-execution events such as op creations and\n      source-file snapshots.\n    op_regex: Dump data from only the tensors from op types that matches to the\n      regular expression (through Python\'s `re.match()`).\n      "Op type" refers to the names of the TensorFlow operations (e.g.,\n      "MatMul", "LogSoftmax"), which may repeat in a TensorFlow\n      function. It does *not* refer to the names of nodes (e.g.,\n      "dense/MatMul", "dense_1/MatMul_1") which are unique within a function.\n      - Example 1: Dump tensor data from only MatMul and Relu ops\n        `op_regex="^(MatMul|Relu)$"`.\n      - Example 2: Dump tensors from all ops *except* Relu:\n        `op_regex="(?!^Relu$)"`.\n      This filter operates in a logical AND relation with `tensor_dtypes`.\n    tensor_dtypes: Dump data from only the tensors of which the specified\n      dtypes. This optional argument can be in any of the following format:\n      - a list or tuple of `DType` objects or strings that can be converted\n        to `DType` objects via `tf.as_dtype()`. Examples:\n        - `tensor_dtype=[tf.float32, tf.float64]`,\n        - `tensor_dtype=["float32", "float64"]`,\n        - `tensor_dtypes=(tf.int32, tf.bool)`,\n        - `tensor_dtypes=("int32", "bool")`\n      - a callable that takes a single `DType` argument and returns a Python\n        `boolean` indicating whether the dtype is to be included in the data\n        dumping. Examples:\n        - `tensor_dtype=lambda dtype: dtype.is_integer`.\n      This filter operates in a logical AND relation with `op_regex`.\n  Returns:\n    A DebugEventsWriter instance used by the dumping callback. The caller\n    may use its flushing methods, including `FlushNonExecutionFiles()` and\n    `FlushExecutionFiles()`.\n  '
    global _state
    tensor_debug_mode_keys = debug_event_pb2.TensorDebugMode.keys()
    if tensor_debug_mode not in tensor_debug_mode_keys:
        raise ValueError("Invalid value in tensor_debug_mode ('%s'). Valid options are: %s" % (tensor_debug_mode, tensor_debug_mode_keys))
    tensor_debug_mode = debug_event_pb2.TensorDebugMode.Value(tensor_debug_mode)
    if tensor_debug_mode not in (debug_event_pb2.TensorDebugMode.NO_TENSOR, debug_event_pb2.TensorDebugMode.CURT_HEALTH, debug_event_pb2.TensorDebugMode.CONCISE_HEALTH, debug_event_pb2.TensorDebugMode.FULL_HEALTH, debug_event_pb2.TensorDebugMode.SHAPE, debug_event_pb2.TensorDebugMode.FULL_TENSOR):
        raise NotImplementedError('tfdbg dumping: support for tensor debug mode %s is not implemented yet' % debug_event_pb2.TensorDebugMode.Name(tensor_debug_mode))
    if tensor_dtypes is not None:
        if not isinstance(tensor_dtypes, (list, tuple)) and (not callable(tensor_dtypes)):
            raise ValueError('If specified, tensor_dtypes is expected to be a list, a tuple, or a callable that takes a DType argument and returns a boolean, but received %s' % (tensor_dtypes,))
        if isinstance(tensor_dtypes, (list, tuple)):
            tensor_dtypes = [dtypes.as_dtype(dtype_item) for dtype_item in tensor_dtypes]
    if hasattr(_state, 'dumping_callback'):
        if _state.dumping_callback.circular_buffer_size != circular_buffer_size:
            raise ValueError('There is already a dumping callback configured with a different circular-buffer size (%d). Therefore the newly request circular-buffer size (%d) will not be honored.' % (_state.dumping_callback.circular_buffer_size, circular_buffer_size))
        if _state.dumping_callback.tensor_debug_mode != tensor_debug_mode:
            raise ValueError('There is already a dumping callback configured for dump root %s with a different tensor-debug mode (%s). Therefore the newly request tensor-debug mode (%s) size will not be honored.' % (_state.dumping_callback.dump_root, tensor_debug_mode_keys[_state.dumping_callback.tensor_debug_mode], tensor_debug_mode_keys[tensor_debug_mode]))
    else:
        _state.dumping_callback = _DumpingCallback(dump_root, tensor_debug_mode, circular_buffer_size, op_regex, tensor_dtypes)
        op_callbacks.add_op_callback(_state.dumping_callback.callback)
        function_lib.CONCRETE_FUNCTION_CALLBACKS.append(_state.dumping_callback.function_callback)
    if _state.dumping_callback.dump_root != dump_root:
        _state.dumping_callback.dump_root = dump_root
    logging.info('Enabled dumping callback in thread %s (dump root: %s, tensor debug mode: %s)', threading.current_thread().name, _state.dumping_callback.dump_root, debug_event_pb2.TensorDebugMode.Name(tensor_debug_mode))
    atexit.register(disable_dump_debug_info)
    return _state.dumping_callback.get_writer()

@tf_export('debugging.experimental.disable_dump_debug_info')
def disable_dump_debug_info():
    if False:
        for i in range(10):
            print('nop')
    'Disable the currently-enabled debugging dumping.\n\n  If the `enable_dump_debug_info()` method under the same Python namespace\n  has been invoked before, calling this method disables it. If no call to\n  `enable_dump_debug_info()` has been made, calling this method is a no-op.\n  Calling this method more than once is idempotent.\n  '
    if hasattr(_state, 'dumping_callback'):
        dump_root = _state.dumping_callback.dump_root
        tfdbg_run_id = _state.dumping_callback.tfdbg_run_id
        debug_events_writer.DebugEventsWriter(dump_root, tfdbg_run_id).Close()
        op_callbacks.remove_op_callback(_state.dumping_callback.callback)
        if _state.dumping_callback.function_callback in function_lib.CONCRETE_FUNCTION_CALLBACKS:
            function_lib.CONCRETE_FUNCTION_CALLBACKS.remove(_state.dumping_callback.function_callback)
        delattr(_state, 'dumping_callback')
        logging.info('Disabled dumping callback in thread %s (dump root: %s)', threading.current_thread().name, dump_root)