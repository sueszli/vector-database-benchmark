from typing import List, Tuple, Union, Dict, Any, Set, Mapping
import collections
from dataclasses import dataclass
import torch
import torch.fx
from torch.fx.node import _get_qualified_name
from torch.fx._compatibility import compatibility
__all__ = ['get_acc_ops_name', 'get_node_target', 'is_node_output_tensor', 'FxNetAccFusionsFinder', 'legalize_graph']
Tensors = Union[Tuple[torch.Tensor], List[torch.Tensor]]
TensorOrTensors = Union[torch.Tensor, Tensors]
NodeList = List[torch.fx.Node]
NodeSet = Set[torch.fx.Node]
Names = List[str]
CALLABLE_NODE_OPS = {'call_module', 'call_function', 'call_method'}

@compatibility(is_backward_compatible=False)
def get_acc_ops_name(k):
    if False:
        while True:
            i = 10
    if isinstance(k, str):
        return k
    elif k.__module__ and 'acc_ops' in k.__module__:
        return f'acc_ops.{k.__name__}'
    else:
        module = k.__module__.replace('torch._ops', 'torch.ops')
        return f"{(module if module else '')}.{k.__name__}"

@compatibility(is_backward_compatible=False)
def get_node_target(submodules: Mapping[str, torch.nn.Module], node: torch.fx.Node) -> str:
    if False:
        return 10
    '\n    Given a `node` returns its target typename.\n\n    For "call_method" node, return node.target which is the name of that method being called.\n    This could potential lead to conflict but should be okay because normally it\'s on a tensor.\n\n    For "call_function" node, return typename of node.target.\n\n    For "call_module" node, return typename of the module that node.target point to.\n\n    If seeing "_VariableFunctionsClass" in the target name string, it will be replaced by\n    "torch". e.g. _VariableFunctionsClass.relu would become torch.relu.\n    '
    assert node.op in CALLABLE_NODE_OPS, 'Expect op types of ' + ', '.join(CALLABLE_NODE_OPS) + f', but found {node.op}'
    if node.op == 'call_module':
        assert isinstance(node.target, str)
        submod = submodules[node.target]
        submod_type = getattr(submod, '_base_class_origin', type(submod))
        return get_acc_ops_name(submod_type)
    elif node.op == 'call_function':
        target: Any = node.target
        return f'acc_ops.{target.__name__}' if target.__module__ is not None and 'acc_ops' in target.__module__ else _get_qualified_name(target)
    else:
        assert isinstance(node.target, str)
        return node.target

@compatibility(is_backward_compatible=False)
def is_node_output_tensor(node: torch.fx.Node) -> bool:
    if False:
        print('Hello World!')
    'Checks if the node output produces a Tensor or not.\n\n    NOTE: This requires to run `ShapeProp` on the containing fx graph before\n    calling this function. This is because it works by checking the `type`\n    metadata on the node. This metadata is produced by the `ShapeProp`.\n    '
    type_ = node.meta.get('type', None)
    return type_ is not None and issubclass(type_, torch.Tensor)

@compatibility(is_backward_compatible=False)
class FxNetAccFusionsFinder:
    """
    Finds groups of connected ACC nodes that pass non-tensor data between each other.
    Such groups are called fusion groups.
    """

    def __init__(self, module: torch.fx.GraphModule, acc_nodes: NodeSet):
        if False:
            while True:
                i = 10
        self.module = module
        self.nodes = list(module.graph.nodes)
        self.acc_nodes = acc_nodes

    @dataclass
    class FusionGroup:
        top_node_idx: int
        nodes: NodeSet
        inputs: NodeSet
        nodes_need_process: NodeSet

        def add_node(self, node):
            if False:
                print('Hello World!')
            '\n            Add a node to fusion group.\n            '
            if node in self.nodes:
                return
            self.nodes_need_process.add(node)
            self.nodes.add(node)
            self.inputs.discard(node)
            self.inputs.update({n for n in node.all_input_nodes if n.op in CALLABLE_NODE_OPS and n not in self.nodes})

    def recursive_add_node(self, fusion_group: 'FxNetAccFusionsFinder.FusionGroup', inputs: Union[NodeSet, NodeList]):
        if False:
            i = 10
            return i + 15
        '\n        Start from inputs and going reverse topological order. If any upstream node\n        is in the fusion group, add all the nodes in this path to fusion group.\n        '
        for arg in inputs:
            if arg.op not in CALLABLE_NODE_OPS:
                continue
            if self.nodes.index(arg) < fusion_group.top_node_idx:
                continue
            if arg in fusion_group.nodes:
                return True
            if self.recursive_add_node(fusion_group, arg.all_input_nodes):
                fusion_group.add_node(arg)
                return True
        return False

    def __call__(self) -> Dict[torch.fx.Node, NodeSet]:
        if False:
            for i in range(10):
                print('nop')
        result: Dict[torch.fx.Node, NodeSet] = {}
        acc_nodes = list(self.acc_nodes)
        for node in acc_nodes:
            if node in result:
                continue
            if node.op not in CALLABLE_NODE_OPS:
                continue
            if 'tensor_meta' in node.meta:
                continue
            if node not in self.acc_nodes:
                continue
            fusion_group: FxNetAccFusionsFinder.FusionGroup = self.FusionGroup(top_node_idx=self.nodes.index(node), nodes={node}, inputs=set(node.all_input_nodes), nodes_need_process={node})
            while fusion_group.nodes_need_process:
                node = fusion_group.nodes_need_process.pop()
                self.recursive_add_node(fusion_group, fusion_group.inputs)
                if 'tensor_meta' not in node.meta:
                    for user in node.users:
                        if user.op not in CALLABLE_NODE_OPS:
                            continue
                        if user in fusion_group.nodes:
                            continue
                        fusion_group.add_node(user)
                        self.recursive_add_node(fusion_group, fusion_group.inputs)
                for arg in node.all_input_nodes:
                    if arg.op not in CALLABLE_NODE_OPS:
                        continue
                    if 'tensor_meta' in arg.meta:
                        continue
                    if arg in fusion_group.nodes:
                        continue
                    fusion_group.add_node(arg)
                    fusion_group.top_node_idx = min(fusion_group.top_node_idx, self.nodes.index(arg))
                    self.recursive_add_node(fusion_group, fusion_group.inputs)
            if not set(fusion_group.nodes) <= self.acc_nodes:
                self.acc_nodes -= fusion_group.nodes
            else:
                for n in fusion_group.nodes:
                    result[n] = fusion_group.nodes
        return result

@compatibility(is_backward_compatible=False)
def legalize_graph(gm: torch.fx.GraphModule) -> torch.fx.GraphModule:
    if False:
        print('Hello World!')
    '\n    Replace the graph of the given GraphModule with one that contains the same nodes as the\n    original, but in topologically sorted order.\n\n    This is used by the merge_matmul transformation below, which disturbs the topologically sorted\n    order of its input GraphModule, so that this order is restored before further transformation.\n\n    Arguments:\n        gm: The graph module to topologically sort. It is modified in-place.\n\n    Returns:\n        The graph module in-place sorted\n    '
    indeg = {node: 0 for node in gm.graph.nodes}
    new_graph = torch.fx.Graph()
    for node in gm.graph.nodes:
        for user in node.users:
            indeg[user] += 1
    queue: collections.deque = collections.deque()
    for node in gm.graph.nodes:
        if indeg[node] == 0:
            queue.append(node)
    env: Dict[torch.fx.Node, torch.fx.Node] = {}
    while len(queue) > 0:
        cur = queue.popleft()
        env[cur] = new_graph.node_copy(cur, lambda x: env[x])
        for user in cur.users:
            indeg[user] -= 1
            if indeg[user] == 0:
                queue.append(user)
    if len(new_graph.nodes) < len(gm.graph.nodes):
        raise RuntimeError(f'Input graph has cycles, unable to add {[node for node in indeg if indeg[node] != 0]}')
    new_graph._codegen = gm.graph._codegen
    gm.graph = new_graph
    return gm