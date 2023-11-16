"""Eager-graph unified check numerics callback."""
import collections
import threading
import numpy as np
from tensorflow.core.protobuf import debug_event_pb2
from tensorflow.python.debug.lib import op_callbacks_common
from tensorflow.python.debug.lib import source_utils
from tensorflow.python.eager import monitoring
from tensorflow.python.framework import op_callbacks
from tensorflow.python.framework import ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import gen_debug_ops
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.util import compat
from tensorflow.python.util import object_identity
from tensorflow.python.util.tf_export import tf_export
IGNORE_OP_OUTPUTS = ((b'FusedBatchNorm', 1), (b'FusedBatchNorm', 2), (b'FusedBatchNorm', 3), (b'FusedBatchNorm', 4), (b'FusedBatchNormV2', 1), (b'FusedBatchNormV2', 2), (b'FusedBatchNormV2', 3), (b'FusedBatchNormV2', 4), (b'FusedBatchNormV3', 1), (b'FusedBatchNormV3', 2), (b'FusedBatchNormV3', 3), (b'FusedBatchNormV3', 4), (b'FusedBatchNormV3', 5))
SAFE_OPS = (b'Concat', b'ConcatV2', b'ExpandDims', b'Fill', b'Gather', b'Maximum', b'Minimum', b'Reshape', b'Slice', b'Squeeze', b'Stack', b'StridedSlice', b'StridedSliceGrad', b'TensorListConcatV2', b'TensorListGather', b'TensorListGetItem', b'TensorListPopBack', b'TensorListStack', b'Transpose', b'Unpack')
_state = threading.local()
_check_numerics_callback_create_counter = monitoring.Counter('/tensorflow/api/python/debugging/check_numerics_callback_create_counter', 'Counter for number of times the check_numerics op callback is created.')

def limit_string_length(string, max_len=50):
    if False:
        print('Hello World!')
    'Limit the length of input string.\n\n  Args:\n    string: Input string.\n    max_len: (int or None) If int, the length limit. If None, no limit.\n\n  Returns:\n    Possibly length-limited string.\n  '
    if max_len is None or len(string) <= max_len:
        return string
    else:
        return '...' + string[len(string) - max_len:]
_CHECK_NUMERICS_INPUT_LOOKUP = collections.defaultdict(dict)

def _maybe_lookup_original_input_tensor(graph, tensor):
    if False:
        return 10
    if graph and graph in _CHECK_NUMERICS_INPUT_LOOKUP and (tensor.name in _CHECK_NUMERICS_INPUT_LOOKUP[graph]):
        return _CHECK_NUMERICS_INPUT_LOOKUP[graph][tensor.name]
    else:
        return tensor

def get_check_numerics_error_message(slot, num_outputs, op_type, tensor, inputs, graph=None, traceback=None, stack_height_limit=30, path_length_limit=50):
    if False:
        while True:
            i = 10
    "Create a meaningful and user-friendly error message about offending tensor.\n\n  The error message reveals the following info about the op that outputs\n  NaN/Infinity: dtype, shape (to the extent known at graph-construction time),\n  input tensors, stack trace for op creation (if is graph mode).\n\n  Args:\n    slot: (int) slot index of the tensor output.\n    num_outputs: (int) total number of outputs of the op.\n    op_type: (str) Type of the that generates `tensor`.\n    tensor: (Tensor) the offending tensor, i.e., the tensor that contains\n      Infinities or NaNs.\n    inputs: (array of Tensor) inputs to the op that generates `tensor`.\n    graph: (tf.Graph) the graph object that `tensor` belongs to. Available only\n      under graph mode.\n    traceback: (list of trace frames) the stack trace of the op's creation.\n      Available only under graph model.\n    stack_height_limit: (int or None) If int, limit to the height of the stack\n      trace printed in the error message. If None, no limit to the height.\n    path_length_limit: (int or None) Length limit for file paths included in the\n      formatted stack trace.\n\n  Returns:\n    (str) A formatted error message.\n  "
    eager_vs_graph_qualifier = 'graph' if graph else 'eagerly-executing'
    message = '\n'
    message += '\n!!! Detected Infinity or NaN in output %d of %s op "%s" (# of outputs: %d) !!!\n' % (slot, eager_vs_graph_qualifier, op_type, num_outputs)
    message += '  dtype: %s\n' % tensor.dtype
    message += '  shape: %s\n' % (tensor.shape,)
    if not graph:
        is_inf = np.isinf(tensor)
        num_neg_inf = np.sum(np.logical_and(np.less(tensor, 0.0), is_inf))
        num_pos_inf = np.sum(np.logical_and(np.greater(tensor, 0.0), is_inf))
        num_nan = np.sum(np.isnan(tensor))
        if num_neg_inf > 0:
            message += '  # of -Inf elements: %s\n' % num_neg_inf
        if num_pos_inf > 0:
            message += '  # of +Inf elements: %s\n' % num_pos_inf
        if num_nan:
            message += '  # of +NaN elements: %s\n' % num_nan
    if len(inputs) > 1:
        message += '\n  Input tensors (%d):\n' % len(inputs)
        for (slot, input_tensor) in enumerate(inputs):
            message += '         %d: %s\n' % (slot, _maybe_lookup_original_input_tensor(graph, input_tensor))
    elif len(inputs) == 1:
        message += '\n  Input tensor: %s\n' % _maybe_lookup_original_input_tensor(graph, inputs[0])
    if graph and hasattr(graph, 'name') and graph.name:
        message += '  Graph name: "%s"\n' % graph.name
    if graph and traceback:
        message += '\n  Stack trace of op\'s creation ("->": inferred user code):\n'
        if stack_height_limit is not None and len(traceback) > stack_height_limit:
            num_omitted_frames = len(traceback) - stack_height_limit
            message += '    + ... (Omitted %d frames)\n' % num_omitted_frames
        for (filepath, lineno, function_name, source_line) in traceback[-stack_height_limit:]:
            user_code_indicator = '    '
            if not source_utils.guess_is_tensorflow_py_library(filepath):
                user_code_indicator = ' -> '
            message += '    + %s (L%d) %s\n' % (limit_string_length(filepath, path_length_limit), lineno, function_name)
            if source_line is not None:
                message += '%s|   %s\n' % (user_code_indicator, source_line)
    message += '\n'
    return message

def _debug_summary(x):
    if False:
        print('Hello World!')
    return gen_debug_ops.debug_numeric_summary_v2(x, tensor_debug_mode=debug_event_pb2.TensorDebugMode.REDUCE_INF_NAN_THREE_SLOTS)

class CheckNumericsCallback(object):
    """Wrapper for the numerics-checking callback for thread locality."""

    def __init__(self, stack_height_limit, path_length_limit):
        if False:
            print('Hello World!')
        self._stack_height_limit = stack_height_limit
        self._path_length_limit = path_length_limit
        self._placeholder_to_debug_tensor = object_identity.ObjectIdentityDictionary()

    def callback(self, op_type, inputs, attrs, outputs, op_name=None, graph=None):
        if False:
            for i in range(10):
                print('nop')
        'Eager-function unified callback for checking numerics.'
        del attrs, op_name
        op_type_bytes = compat.as_bytes(op_type)
        is_v1_graph_mode = not ops.executing_eagerly_outside_functions()
        if op_type_bytes in op_callbacks_common.OP_CALLBACK_SKIP_OPS or op_type_bytes in SAFE_OPS:
            return None
        if graph:
            instrumented_outputs = []
            if is_v1_graph_mode:
                for input_tensor in inputs:
                    if input_tensor in self._placeholder_to_debug_tensor and outputs:
                        outputs[0].op._add_control_input(self._placeholder_to_debug_tensor[input_tensor].op)
            for (slot, output) in enumerate(outputs):
                if output.dtype.is_floating and (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS:
                    checked_output = array_ops.check_numerics_v2(output if is_v1_graph_mode else _debug_summary(output), get_check_numerics_error_message(slot, len(outputs), op_type, output, inputs, graph=graph, traceback=output.op.traceback, stack_height_limit=self._stack_height_limit, path_length_limit=self._path_length_limit))
                    _CHECK_NUMERICS_INPUT_LOOKUP[graph][checked_output.name] = output
                    instrumented_outputs.append(self._get_output_tensor(op_type_bytes, output, checked_output, is_v1_graph_mode))
                else:
                    instrumented_outputs.append(output)
            return instrumented_outputs
        else:
            if op_type_bytes == b'CheckNumericsV2':
                return None
            for (slot, output) in enumerate(outputs):
                if output.dtype.is_floating and (op_type_bytes, slot) not in IGNORE_OP_OUTPUTS:
                    array_ops.check_numerics_v2(output, get_check_numerics_error_message(slot, len(outputs), op_type, output, inputs, stack_height_limit=self._stack_height_limit, path_length_limit=self._path_length_limit))

    def _get_output_tensor(self, op_type, tensor, checked_tensor, is_v1_graph_mode):
        if False:
            for i in range(10):
                print('nop')
        'Determine what tensor to output from callback.\n\n    Args:\n      op_type: Type of the op that outputs the original symbolic tensor, as\n        `bytes`.\n      tensor: The original output symbolic tensor.\n      checked_tensor: The debugger-instrumented, numerics-checking tensor.\n      is_v1_graph_mode: Whether the debugged proggram is running under V1 graph\n        mode.\n\n    Returns:\n      A symbolic tensor to be returned by the dumping op_callback.\n    '
        if is_v1_graph_mode:
            if op_type == b'Placeholder':
                self._placeholder_to_debug_tensor[tensor] = checked_tensor
                return tensor
            else:
                return checked_tensor
        else:
            return tensor

@tf_export('debugging.enable_check_numerics')
def enable_check_numerics(stack_height_limit=30, path_length_limit=50):
    if False:
        i = 10
        return i + 15
    'Enable tensor numerics checking in an eager/graph unified fashion.\n\n  The numerics checking mechanism will cause any TensorFlow eager execution or\n  graph execution to error out as soon as an op\'s output tensor contains\n  infinity or NaN.\n\n  This method is idempotent. Calling it multiple times has the same effect\n  as calling it once.\n\n  This method takes effect only on the thread in which it is called.\n\n  When a op\'s float-type output tensor contains any Infinity or NaN, an\n  `tf.errors.InvalidArgumentError` will be thrown, with an error message that\n  reveals the following information:\n    - The type of the op that generated the tensor with bad numerics.\n    - Data type (dtype) of the tensor.\n    - Shape of the tensor (to the extent known at the time of eager execution\n      or graph construction).\n    - Name of the containing graph (if available).\n    - (Graph mode only): The stack trace of the intra-graph op\'s creation,\n      with a stack-height limit and a path-length limit for visual clarity.\n      The stack frames that belong to the user\'s code (as opposed to\n      tensorflow\'s internal code) are highlighted with a text arrow ("->").\n    - (Eager mode only): How many of the offending tensor\'s elements are\n      `Infinity` and `NaN`, respectively.\n\n  Once enabled, the check-numerics mechanism can be disabled by using\n  `tf.debugging.disable_check_numerics()`.\n\n  Example usage:\n\n  1. Catching infinity during the execution of a `tf.function` graph:\n\n     ```py\n     import tensorflow as tf\n\n     tf.debugging.enable_check_numerics()\n\n     @tf.function\n     def square_log_x_plus_1(x):\n       v = tf.math.log(x + 1)\n       return tf.math.square(v)\n\n     x = -1.0\n\n     # When the following line runs, a function graph will be compiled\n     # from the Python function `square_log_x_plus_1()`. Due to the\n     # `enable_check_numerics()` call above, the graph will contain\n     # numerics checking ops that will run during the function graph\'s\n     # execution. The function call generates an -infinity when the Log\n     # (logarithm) op operates on the output tensor of the Add op.\n     # The program errors out at this line, printing an error message.\n     y = square_log_x_plus_1(x)\n     z = -y\n    ```\n\n  2. Catching NaN during eager execution:\n\n     ```py\n     import numpy as np\n     import tensorflow as tf\n\n     tf.debugging.enable_check_numerics()\n\n     x = np.array([[0.0, -1.0], [4.0, 3.0]])\n\n     # The following line executes the Sqrt op eagerly. Due to the negative\n     # element in the input array, a NaN is generated. Due to the\n     # `enable_check_numerics()` call above, the program errors immediately\n     # at this line, printing an error message.\n     y = tf.math.sqrt(x)\n     z = tf.matmul(y, y)\n     ```\n\n  NOTE: If your code is running on TPUs, be sure to call\n  `tf.config.set_soft_device_placement(True)` before calling\n  `tf.debugging.enable_check_numerics()` as this API uses automatic outside\n  compilation on TPUs. For example:\n\n  ```py\n  tf.config.set_soft_device_placement(True)\n  tf.debugging.enable_check_numerics()\n\n  resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=\'\')\n  strategy = tf.distribute.TPUStrategy(resolver)\n  with strategy.scope():\n    # ...\n  ```\n\n  Args:\n    stack_height_limit: Limit to the height of the printed stack trace.\n      Applicable only to ops in `tf.function`s (graphs).\n    path_length_limit: Limit to the file path included in the printed stack\n      trace. Applicable only to ops in `tf.function`s (graphs).\n  '
    if not hasattr(_state, 'check_numerics_callback'):
        _state.check_numerics_callback = CheckNumericsCallback(stack_height_limit, path_length_limit)
    op_callbacks.add_op_callback(_state.check_numerics_callback.callback)
    logging.info('Enabled check-numerics callback in thread %s', threading.current_thread().name)
    _check_numerics_callback_create_counter.get_cell().increase_by(1)

@tf_export('debugging.disable_check_numerics')
def disable_check_numerics():
    if False:
        for i in range(10):
            print('nop')
    'Disable the eager/graph unified numerics checking mechanism.\n\n  This method can be used after a call to `tf.debugging.enable_check_numerics()`\n  to disable the numerics-checking mechanism that catches infinity and NaN\n  values output by ops executed eagerly or in tf.function-compiled graphs.\n\n  This method is idempotent. Calling it multiple times has the same effect\n  as calling it once.\n\n  This method takes effect only on the thread in which it is called.\n  '
    if not hasattr(_state, 'check_numerics_callback'):
        return
    try:
        op_callbacks.remove_op_callback(_state.check_numerics_callback.callback)
        delattr(_state, 'check_numerics_callback')
        logging.info('Disabled check-numerics callback in thread %s', threading.current_thread().name)
    except KeyError:
        pass