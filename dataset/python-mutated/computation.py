import torch._C._lazy
import torch._C._lazy_ts_backend

def get_tensors_ts_device_data_node(tensors):
    if False:
        print('Hello World!')
    'Return tensor ids and eager tensors for DeviceData nodes in the\n    IR for the passed in lazy tensors.\n\n    TODO: This API is currently ts backend specific. We are working on\n    generalizing it to all backends including XLA.\n    '
    return torch._C._lazy_ts_backend._get_tensors_ts_device_data_node(tensors)

def get_graph_hash(tensors):
    if False:
        i = 10
        return i + 15
    'Return the graph hash for the passed in lazy tensors'
    return torch._C._lazy._get_graph_hash(tensors)

def run_cached_graph(hash_str, graph_inputs):
    if False:
        return 10
    'Running the cached computation graph with the given inputs\n\n    TODO: This API is currently ts backend specific. We are working on\n    generalizing it to all backends including XLA.\n    '
    return torch._C._lazy_ts_backend._run_cached_graph(hash_str, graph_inputs)