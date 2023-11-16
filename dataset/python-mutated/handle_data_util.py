"""Decorator to overrides the gradient for a function."""
from tensorflow.python.client import pywrap_tf_session
from tensorflow.python.framework import cpp_shape_inference_pb2
from tensorflow.python.framework import dtypes
from tensorflow.python.types import core
from tensorflow.python.util import compat

def get_resource_handle_data(graph_op):
    if False:
        while True:
            i = 10
    assert isinstance(graph_op, core.Symbol) and (not isinstance(graph_op, core.Value))
    with graph_op.graph._c_graph.get() as c_graph:
        handle_data = pywrap_tf_session.GetHandleShapeAndType(c_graph, graph_op._as_tf_output())
    return cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData.FromString(compat.as_bytes(handle_data))

def get_handle_data(source_t):
    if False:
        for i in range(10):
            print('nop')
    'Obtains HandleData from a tensor.'
    if isinstance(source_t, core.Value):
        return source_t._handle_data
    return get_resource_handle_data(source_t)

def copy_handle_data(source_t, target_t):
    if False:
        for i in range(10):
            print('nop')
    "Copies HandleData for variant and resource type tensors if available.\n\n  The CppShapeInferenceResult::HandleData proto contains information about the\n  shapes and types of the element tensors of resource/variant type tensors.\n  We need to copy this across function boundaries, i.e., when capturing a\n  placeholder or when returning a function tensor as output. If we don't do this\n  the element tensors will have unknown shapes, e.g., if a TensorList variant\n  tensor is captured as a placeholder, elements popped from that list would have\n  unknown shape.\n\n  Args:\n    source_t: The tensor to copy HandleData from.\n    target_t: The tensor to copy HandleData to.\n  "
    if target_t.dtype == dtypes.resource or target_t.dtype == dtypes.variant:
        handle_data = get_handle_data(source_t)
        set_handle_data(target_t, handle_data)

def set_handle_data(target_t, handle_data):
    if False:
        print('Hello World!')
    'Sets handle data on the giver tensor.'
    if handle_data is None or not handle_data.is_set or (not handle_data.shape_and_type):
        return
    if isinstance(target_t, core.Value):
        target_t._handle_data = handle_data
        return
    with target_t.graph._c_graph.get() as c_graph:
        pywrap_tf_session.SetHandleShapeAndType(c_graph, target_t._as_tf_output(), handle_data.SerializeToString())

def create_handle_data(shape, dtype):
    if False:
        i = 10
        return i + 15
    handle_data = cpp_shape_inference_pb2.CppShapeInferenceResult.HandleData()
    handle_data.is_set = True
    handle_data.shape_and_type.append(cpp_shape_inference_pb2.CppShapeInferenceResult.HandleShapeAndType(shape=shape.as_proto(), dtype=dtype.as_datatype_enum))
    return handle_data