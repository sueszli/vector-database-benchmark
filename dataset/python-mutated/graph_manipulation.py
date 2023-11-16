from typing import Any, Dict, List, NamedTuple, Optional
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph import Graph
from torch.fx.graph_module import GraphModule
from torch.fx.node import map_arg, Node, Target
from torch.fx.passes.shape_prop import ShapeProp
__all__ = ['replace_target_nodes_with', 'size_bytes', 'get_size_of_all_nodes', 'get_tensor_meta', 'get_size_of_node']

@compatibility(is_backward_compatible=False)
def replace_target_nodes_with(fx_module: GraphModule, old_op: str, old_target: Target, new_op: str, new_target: Target):
    if False:
        i = 10
        return i + 15
    'Modifies all nodes in fx_module.graph.nodes which match the specified op code and target,\n    and updates them to match the new op code and target'
    new_graph = Graph()
    val_map: Dict[Node, Node] = {}
    for node in fx_module.graph.nodes:
        if node.op == old_op and node.target == old_target:
            args = map_arg(node.args, lambda n: val_map[n])
            kwargs = map_arg(node.kwargs, lambda n: val_map[n])
            assert isinstance(args, tuple)
            assert isinstance(kwargs, dict)
            val_map[node] = new_graph.create_node(new_op, new_target, args, kwargs, node.name)
        else:
            val_map[node] = new_graph.node_copy(node, lambda n: val_map[n])
    fx_module.graph = new_graph

@compatibility(is_backward_compatible=False)
class size_bytes(NamedTuple):
    output_size: int
    total_size: int

@compatibility(is_backward_compatible=False)
def get_size_of_all_nodes(fx_module: GraphModule, args: Optional[List[torch.Tensor]]=None) -> None:
    if False:
        print('Hello World!')
    'Given a fx graph module, update each node with its total size (weights + bias + output)\n    and its output_size(output). For a non-module node, the total size is the output size.\n    return total size'
    if args is not None:
        ShapeProp(fx_module).propagate(*args)
    total_size_of_graph = 0.0
    for node in fx_module.graph.nodes:
        if node.op == 'output':
            break
        node.size_bytes = get_size_of_node(fx_module, node)
    return

@compatibility(is_backward_compatible=False)
def get_tensor_meta(node: Node) -> Any:
    if False:
        while True:
            i = 10
    tensor_meta = node.meta.get('tensor_meta')
    if not tensor_meta:
        raise RuntimeError(f'Node {node} has no tensor metadata associated with it! Check that shape propagation has run.')
    return tensor_meta

@compatibility(is_backward_compatible=False)
def get_size_of_node(fx_module: GraphModule, node: Node) -> size_bytes:
    if False:
        print('Hello World!')
    'Given a node with node.dtype and node.shape, return its total size and its output size.\n    total_size = weights + bias + output_size\n    '
    total_num_of_elems = 0
    if node.op == 'call_module':
        submodule_dict = dict(fx_module.named_modules())
        submodule = submodule_dict[node.target]
        parameters = submodule.named_parameters()
        for (name, p) in parameters:
            total_num_of_elems += p.numel()
    tensor_meta = get_tensor_meta(node)
    output_elem = tensor_meta.shape.numel()
    total_num_of_elems += output_elem
    if tensor_meta.is_quantized:
        size_per_elem_bytes = torch._empty_affine_quantized([], dtype=tensor_meta.dtype).element_size()
    else:
        size_per_elem_bytes = torch.tensor([], dtype=tensor_meta.dtype).element_size()
    total_size = size_per_elem_bytes * total_num_of_elems
    output_size = size_per_elem_bytes * output_elem
    return size_bytes(output_size, total_size)