"""Custom Aggregator op is for collecting numeric metrics from the given input."""
from tensorflow.compiler.mlir.quantization.tensorflow.calibrator import custom_aggregator_op_wrapper
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import load_library
from tensorflow.python.platform import resource_loader
_custom_aggregator_op = load_library.load_op_library(resource_loader.get_path_to_datafile('_custom_aggregator_op.so'))

def custom_aggregator(input_tensor, tensor_id: str):
    if False:
        i = 10
        return i + 15
    'Creates custom aggregator op that collects numeric metrics from the tensor.\n\n  Args:\n    input_tensor: Tensor to be scanned through this operator. This tensor will\n      be bypassed to the output tensor of this operator.\n    tensor_id: String, the identity of the tensor to be scanned.\n\n  Returns:\n    A `Tensor` of the same value as `input_tensor`.\n\n  Raises:\n    ValueError: If the given type of `input_tensor` is not float32.\n  '
    if input_tensor.dtype != dtypes.float32:
        raise ValueError('Custom aggregator op only accept float32 values.')
    return custom_aggregator_op_wrapper.custom_aggregator(input_tensor, tensor_id)