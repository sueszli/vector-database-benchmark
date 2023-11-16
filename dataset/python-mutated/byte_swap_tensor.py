"""Utilities for byte swapping the tensor content."""
from tensorflow.core.framework import graph_pb2
from tensorflow.core.protobuf import meta_graph_pb2
from tensorflow.python.framework import dtypes
byte_swappable = [dtypes.float16, dtypes.float32, dtypes.float64, dtypes.bfloat16, dtypes.complex64, dtypes.complex128, dtypes.uint16, dtypes.uint32, dtypes.uint64, dtypes.int16, dtypes.int32, dtypes.int64, dtypes.qint16, dtypes.quint16, dtypes.qint32]

def byte_swap_tensor_content(tensor, from_endiness, to_endiness):
    if False:
        return 10
    'Byte swaps.\n\n  Args:\n    tensor: Target tensor to change endiness.\n    from_endiness: The original endianness format. "big" or "little"\n    to_endiness: The target endianness format. "big" or "little"\n  '
    if tensor.dtype in byte_swappable:
        tshape = tensor.tensor_shape.dim
        tensor_bytes = tensor.tensor_content
        if tensor_bytes:
            tensor_size = 1
            for sz in tshape:
                if sz.size != 0:
                    tensor_size = tensor_size * sz.size
            chunksize = int(len(tensor_bytes) / tensor_size)
            to_swap = [tensor_bytes[i:i + chunksize] for i in range(0, len(tensor_bytes), chunksize)]
            tensor.tensor_content = b''.join([int.from_bytes(byteswap, from_endiness).to_bytes(chunksize, to_endiness) for byteswap in to_swap])

def swap_tensor_content_in_graph_function(graph_def, from_endiness, to_endiness):
    if False:
        i = 10
        return i + 15
    'Fix endiness of tensor contents.\n\n  Args:\n    graph_def: Target graph_def to change endiness.\n    from_endiness: The original endianness format. "big" or "little"\n    to_endiness: The target endianness format. "big" or "little"\n  '
    if isinstance(graph_def, meta_graph_pb2.MetaGraphDef):
        functions = graph_def.graph_def.library.function
    elif isinstance(graph_def, graph_pb2.GraphDef):
        functions = graph_def.library.function
    else:
        return
    for function in functions:
        node_def = function.node_def
        for node in node_def:
            if node.op == 'Const':
                tensor = node.attr['value'].tensor
                byte_swap_tensor_content(tensor, from_endiness, to_endiness)

def swap_tensor_content_in_graph_node(graph_def, from_endiness, to_endiness):
    if False:
        print('Hello World!')
    for node in graph_def.node:
        if node.op == 'Const':
            tensor = node.attr['value'].tensor
            byte_swap_tensor_content(tensor, from_endiness, to_endiness)