import copy
from queue import SimpleQueue
from typing import List, Dict, Tuple
import torch.fx
from torch.fx.graph_module import GraphModule
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx.passes.tools_common import NodeList, NodeSet, legalize_graph
from torch.fx.passes.utils import lift_subgraph_as_module
from torch.fx._compatibility import compatibility

@compatibility(is_backward_compatible=False)
def topo_sort(nodes: NodeList) -> NodeList:
    if False:
        print('Hello World!')
    indegree_map = {node: 0 for node in nodes}
    candidates: SimpleQueue = SimpleQueue()
    for node in nodes:
        for n in node.all_input_nodes:
            if n in indegree_map:
                indegree_map[node] += 1
        if indegree_map[node] == 0:
            candidates.put(node)
    sorted_nodes: NodeList = list()
    while not candidates.empty():
        node = candidates.get()
        sorted_nodes.append(node)
        for n in node.users:
            if n in indegree_map:
                indegree_map[n] -= 1
                if indegree_map[n] == 0:
                    candidates.put(n)
    assert len(nodes) == len(sorted_nodes), "topological sorted nodes doesn't have same length as input nodes"
    return sorted_nodes

@compatibility(is_backward_compatible=False)
def validate_partition(partition: NodeList) -> bool:
    if False:
        i = 10
        return i + 15
    partition_set = set(partition)
    outputs: NodeList = list()
    for node in partition_set:
        for user_node in node.users:
            if user_node not in partition_set:
                outputs.append(user_node)

    def bfs_find_cycle(root_nodes: NodeList) -> bool:
        if False:
            while True:
                i = 10
        visited: NodeSet = set()
        queue: NodeList = root_nodes
        while queue:
            current = queue.pop()
            visited.add(current)
            if current in partition_set:
                return True
            for user_node in current.users:
                if user_node in visited:
                    continue
                queue.append(user_node)
        return False
    if bfs_find_cycle(outputs):
        return False
    return True

@compatibility(is_backward_compatible=False)
def fuse_as_graphmodule(gm: GraphModule, nodes: NodeList, module_name: str) -> Tuple[GraphModule, Tuple[Node, ...], Tuple[Node, ...]]:
    if False:
        while True:
            i = 10
    '\n    Fuse nodes in graph_module into a GraphModule.\n\n    Args:\n        gm (GraphModule): target graph_module\n\n        nodes (List[Node]): list of nodes in `gm` to fuse, where the node must be topologically sorted\n\n        module_name: class name for the fused GraphModule\n\n    Returns:\n        fused_gm (GraphModule): fused graph module, where its node is a copy of `nodes` in `gm`\n\n        original_inputs (Tuple[Node, ...]): input nodes to `nodes` in original `gm`\n\n        original_outputs (Tuple[Node, ...]): consumer nodes of `nodes` in original `gm`\n\n    '
    for node in nodes:
        assert node.graph.owning_module is gm, f"{node} doesn't belong to passed in graph module {gm._get_name()}"
        assert not node._erased, f'{node} has been removed from owning graph'
        assert node in gm.graph.nodes, f'{node} is not found in graph module {gm._get_name()}'
    assert validate_partition(nodes), 'Invalid partition, found dependency cycles'
    subgraph = Graph()
    node_to_placeholder: Dict[Node, Node] = {}
    node_map: Dict[Node, Node] = {}

    def remap_inputs(x):
        if False:
            print('Hello World!')
        if x.op == 'get_attr':
            pass
        if x in nodes:
            return node_map[x]
        if x not in node_to_placeholder:
            placeholder_node = subgraph.placeholder(x.name, type_expr=x.type)
            placeholder_node.meta = copy.copy(x.meta)
            node_to_placeholder[x] = placeholder_node
        return node_to_placeholder[x]
    for node in nodes:
        new_node = subgraph.node_copy(node, remap_inputs)
        node_map[node] = new_node
    output_mapping: Dict[Node, Node] = {}
    for node in nodes:
        for user_node in node.users:
            if user_node not in nodes:
                output_mapping[node] = node_map[node]
    outs = tuple(output_mapping.values())
    subgraph.output(outs[0] if len(outs) == 1 else outs)
    subgraph.lint()
    fused_gm: GraphModule
    (fused_gm, _) = lift_subgraph_as_module(gm, subgraph, comp_name='', class_name=module_name)
    original_inputs: Tuple[Node, ...] = tuple(node_to_placeholder.keys())
    original_outputs: Tuple[Node, ...] = tuple(output_mapping.keys())
    return (fused_gm, original_inputs, original_outputs)

@compatibility(is_backward_compatible=False)
def insert_subgm(gm: GraphModule, sub_gm: GraphModule, orig_inputs: Tuple[Node, ...], orig_outputs: Tuple[Node, ...]):
    if False:
        for i in range(10):
            print('nop')
    submodule_name = sub_gm.__class__.__name__
    gm.add_submodule(submodule_name, sub_gm)
    module_node = gm.graph.call_module(submodule_name, args=orig_inputs, kwargs=None)
    if len(orig_outputs) == 1:
        orig_outputs[0].replace_all_uses_with(module_node, propagate_meta=True)
    else:
        for (i, orig_output) in enumerate(orig_outputs):
            proxy_out = torch.fx.Proxy(module_node)[i].node
            orig_output.replace_all_uses_with(proxy_out, propagate_meta=True)
    return gm

@compatibility(is_backward_compatible=False)
def erase_nodes(gm: GraphModule, nodes: NodeList):
    if False:
        return 10
    for node in reversed(nodes):
        gm.graph.erase_node(node)

@compatibility(is_backward_compatible=False)
def fuse_by_partitions(gm: GraphModule, partitions: List[NodeList]) -> GraphModule:
    if False:
        while True:
            i = 10
    for (partition_id, nodes) in enumerate(partitions):
        sorted_nodes = topo_sort(nodes)
        submodule_name = 'fused_' + str(partition_id)
        (sub_gm, orig_inputs, orig_outputs) = fuse_as_graphmodule(gm, sorted_nodes, submodule_name)
        insert_subgm(gm, sub_gm, orig_inputs, orig_outputs)
        erase_nodes(gm, sorted_nodes)
    legalize_graph(gm)
    return gm