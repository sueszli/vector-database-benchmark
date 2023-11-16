import inspect
from typing import Any, Callable, Dict, List, Optional
import torch
from torch.fx._compatibility import compatibility
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node
__all__ = ['Partition', 'split_module']

@compatibility(is_backward_compatible=True)
class Partition:

    def __init__(self, name: str):
        if False:
            for i in range(10):
                print('nop')
        self.name: str = name
        self.submod_name = f'submod_{name}'
        self.node_names: List[str] = []
        self.inputs: Dict[str, None] = {}
        self.outputs: Dict[str, None] = {}
        self.dependencies: Dict[str, None] = {}
        self.dependents: Dict[str, None] = {}
        self.graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
        self.environment: Dict[Node, Node] = {}
        self.targets: Dict[str, Any] = {}

    def __repr__(self) -> str:
        if False:
            i = 10
            return i + 15
        return f'name: {self.name},\n nodes: {self.node_names},\n inputs: {self.inputs},\n outputs: {self.outputs},\n partitions depended on: {self.dependencies},\n partition dependents: {self.dependents}'

@compatibility(is_backward_compatible=True)
def split_module(m: GraphModule, root_m: torch.nn.Module, split_callback: Callable[[Node], int], qualname_map: Optional[Dict[str, str]]=None, keep_original_order: Optional[bool]=False):
    if False:
        i = 10
        return i + 15
    '\n    Creates subgraphs out of main graph\n\n    Args:\n        m (GraphModule): Graph module to split\n        root_m (torch.nn.Module): root nn module. Not currently used. Included\n            because the root nn module is usually transformed via\n            torch.fx._symbolic_trace.symbolic_trace (see example below)\n        split_callback (Callable[[Node], int]): Callable function\n            that maps a given Node instance to a numeric partition identifier.\n            split_module will use this function as the policy for which operations\n            appear in which partitions in the output Module.\n        qualname_map: Optional[Dict[str, str]]: optional output parameter that returns a\n            mapping from new target names in the module after split to old target\n            names in the original module.\n        keep_original_order: Optional[bool]: keep the original order of the GraphModule\n            or use the Topological order of the new constructed GraphModule\n\n\n    Returns:\n        GraphModule: the module after split.\n\n    Example:\n\n        This is a sample setup:\n\n            import torch\n            from torch.fx.symbolic_trace import symbolic_trace\n            from torch.fx.graph_module import GraphModule\n            from torch.fx.node import Node\n            from torch.fx.passes.split_module import split_module\n\n            class MyModule(torch.nn.Module):\n                def __init__(self):\n                    super().__init__()\n                    self.param = torch.nn.Parameter(torch.rand(3, 4))\n                    self.linear = torch.nn.Linear(4, 5)\n\n                def forward(self, x, y):\n                    z = self.linear(x + self.param).clamp(min=0.0, max=1.0)\n                    w = self.linear(y).clamp(min=0.0, max=1.0)\n                    return z + w\n\n            # symbolically trace model\n            my_module = MyModule()\n            my_module_traced = symbolic_trace(my_module)\n\n            # random mod partitioning\n            partition_counter = 0\n            NPARTITIONS = 3\n\n            def mod_partition(node: Node):\n                global partition_counter\n                partition = partition_counter % NPARTITIONS\n                partition_counter = (partition_counter + 1) % NPARTITIONS\n                return partition\n\n            # split module in module with submodules\n            module_with_submodules = split_module(\n                my_module_traced, my_module, mod_partition\n            )\n\n        Output looks like this. Original graph is broken into partitions\n\n            > print(module_with_submodules)\n            GraphModule(\n                (submod_0): GraphModule(\n                    (linear): Linear(in_features=4, out_features=5, bias=True)\n                )\n                (submod_1): GraphModule(\n                    (linear): Linear(in_features=4, out_features=5, bias=True)\n                )\n                (submod_2): GraphModule()\n            )\n\n            def forward(self, x, y):\n                param = self.param\n                submod_0 = self.submod_0(x, param, y);  x = param = y = None\n                getitem = submod_0[0]\n                getitem_1 = submod_0[1];  submod_0 = None\n                submod_1 = self.submod_1(getitem, getitem_1);  getitem = getitem_1 = None\n                getitem_2 = submod_1[0]\n                getitem_3 = submod_1[1];  submod_1 = None\n                submod_2 = self.submod_2(getitem_2, getitem_3);  getitem_2 = getitem_3 = None\n                return submod_2\n\n        Output of split module is the same as output of input traced module.\n        This is an example within a test setting:\n\n            > orig_out = my_module_traced(x, y)\n            > submodules_out = module_with_submodules(x, y)\n            > self.assertEqual(orig_out, submodules_out)\n            True\n    '

    def construct_graph(node: Node, base_mod_env: Dict[str, Node], base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule]):
        if False:
            print('Hello World!')
        if node.op == 'placeholder':
            default_value = node.args[0] if len(node.args) > 0 else inspect.Signature.empty
            base_mod_env[node.name] = base_mod_graph.placeholder(node.target, type_expr=node.type, default_value=default_value)
            base_mod_env[node.name].meta = node.meta.copy()
        elif node.op == 'get_attr':
            base_mod_env[node.name] = base_mod_graph.get_attr(node.target)
            base_mod_env[node.name].meta = node.meta.copy()
            attr_val = m
            for atom in node.target.split('.'):
                if not hasattr(attr_val, atom):
                    raise AttributeError(f'Node target {node.target} not found!')
                attr_val = getattr(attr_val, atom)
            base_mod_attrs[node.target] = attr_val
        return (base_mod_env, base_mod_attrs)
    partitions: Dict[str, Partition] = {}
    orig_nodes: Dict[str, Node] = {}

    def record_cross_partition_use(def_node: Node, use_node: Optional[Node]):
        if False:
            return 10
        defined = getattr(def_node, '_fx_partition', None)
        used = getattr(use_node, '_fx_partition', None)
        if defined != used:
            if defined is not None:
                def_partition = partitions[defined]
                def_partition.outputs.setdefault(def_node.name)
                if used is not None:
                    def_partition.dependents.setdefault(used)
            if used is not None:
                use_partition = partitions[used]
                use_partition.inputs.setdefault(def_node.name)
                if defined is not None:
                    use_partition.dependencies.setdefault(defined)

    def instantiate_node_partition_mapping(node):
        if False:
            while True:
                i = 10
        partition_name = str(split_callback(node))
        partition = partitions.get(partition_name)
        if partition is None:
            partitions[partition_name] = partition = Partition(partition_name)
        partition.node_names.append(node.name)
        node._fx_partition = partition_name
    for node in m.graph.nodes:
        orig_nodes[node.name] = node
        if node.op in ['placeholder', 'get_attr']:
            continue
        if node.op == 'output':
            torch.fx.graph.map_arg(node.args[0], lambda n: record_cross_partition_use(n, None))
            continue
        instantiate_node_partition_mapping(node)
        torch.fx.graph.map_arg(node.args, lambda def_node: record_cross_partition_use(def_node, node))
        torch.fx.graph.map_arg(node.kwargs, lambda def_node: record_cross_partition_use(def_node, node))
    original_partition_order = list(partitions.keys())
    root_partitions: List[str] = []
    for (partition_name, partition) in partitions.items():
        if not len(partition.dependencies):
            root_partitions.append(partition_name)
    sorted_partitions: List[str] = []
    while root_partitions:
        root_partition = root_partitions.pop()
        sorted_partitions.append(root_partition)
        for dependent in partitions[root_partition].dependents:
            partitions[dependent].dependencies.pop(root_partition)
            if not partitions[dependent].dependencies:
                root_partitions.append(dependent)
    if len(sorted_partitions) != len(partitions):
        raise RuntimeError('cycle exists between partitions!')
    for partition_name in sorted_partitions:
        partition = partitions[partition_name]
        for inp in partition.inputs:
            placeholder = partition.graph.placeholder(inp, type_expr=orig_nodes[inp].type)
            placeholder.meta = orig_nodes[inp].meta.copy()
            partition.environment[orig_nodes[inp]] = placeholder
    for node in m.graph.nodes:
        if hasattr(node, '_fx_partition'):
            partition = partitions[node._fx_partition]
            environment = partition.environment
            gathered_args = torch.fx.graph.map_arg(node.args, lambda n: environment[n])
            gathered_kwargs = torch.fx.graph.map_arg(node.kwargs, lambda n: environment[n])
            if node.op not in ['call_module', 'get_attr']:
                target = node.target
            else:
                target_atoms = node.target.split('.')
                target_attr = m
                for atom in target_atoms:
                    if not hasattr(target_attr, atom):
                        raise AttributeError(f'Operator target {node.target} not found!')
                    target_attr = getattr(target_attr, atom)
                target = '_'.join(target_atoms)
                partition.targets[target] = target_attr
                if qualname_map is not None:
                    qualname = f'{partition.submod_name}.{target}'
                    qualname_map[qualname] = node.target
            assert isinstance(gathered_args, tuple)
            assert isinstance(gathered_kwargs, dict)
            new_node = partition.graph.create_node(op=node.op, target=target, args=gathered_args, kwargs=gathered_kwargs, type_expr=node.type)
            new_node.meta = node.meta.copy()
            partition.environment[node] = new_node
    orig_mod_env: Dict[str, Node] = {}
    base_mod_env: Dict[str, Node] = {}
    base_mod_graph: torch.fx.graph.Graph = torch.fx.graph.Graph()
    base_mod_attrs: Dict[str, torch.fx.graph_module.GraphModule] = {}
    if not keep_original_order:
        for node in m.graph.nodes:
            (base_mod_env, base_mod_attrs) = construct_graph(node, base_mod_env, base_mod_attrs)
    else:
        for node in m.graph.nodes:
            orig_mod_env[node.name] = node
    construct_order_partitions = sorted_partitions if not keep_original_order else original_partition_order
    already_constructed_attr_nodes = set()
    for partition_name in construct_order_partitions:
        partition = partitions[partition_name]
        output_vals = tuple((partition.environment[orig_nodes[name]] for name in partition.outputs))
        num_output_vals = len(output_vals)
        if num_output_vals == 1:
            partition.graph.output(output_vals[0])
        elif num_output_vals > 1:
            partition.graph.output(output_vals)
        if keep_original_order:
            orig_mod_attr_nodes: List[Node] = [orig_mod_env[key] for key in partition.inputs]
            for node in orig_mod_attr_nodes:
                if node in already_constructed_attr_nodes:
                    continue
                (base_mod_env, base_mod_attrs) = construct_graph(node, base_mod_env, base_mod_attrs)
                already_constructed_attr_nodes.add(node)
        base_mod_attrs[partition.submod_name] = torch.fx.graph_module.GraphModule(partition.targets, partition.graph)
        output_val = base_mod_graph.call_module(partition.submod_name, tuple((base_mod_env[name] for name in partition.inputs)))
        num_outputs = len(partition.outputs)
        if num_outputs > 1:
            output_val_proxy = torch.fx.proxy.Proxy(output_val)
            for (i, output_name) in enumerate(partition.outputs):
                base_mod_env[output_name] = output_val_proxy[i].node
        elif num_outputs == 1:
            base_mod_env[next(iter(partition.outputs))] = output_val
    for node in m.graph.nodes:
        if node.op == 'output':
            base_mod_graph.output(torch.fx.graph.map_arg(node.args[0], lambda n: base_mod_env[n.name]))
    return torch.fx.graph_module.GraphModule(base_mod_attrs, base_mod_graph)