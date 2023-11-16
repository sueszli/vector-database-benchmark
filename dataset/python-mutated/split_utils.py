import copy
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Type, Union
import torch.fx
from torch.fx._compatibility import compatibility
from torch.fx.graph import map_arg
from torch.fx.passes.utils import HolderModule, lift_subgraph_as_module
from .tools_common import NodeList
__all__ = ['getattr_recursive', 'setattr_recursive', 'Component', 'split_by_tags']

@compatibility(is_backward_compatible=False)
def getattr_recursive(obj, name):
    if False:
        i = 10
        return i + 15
    for layer in name.split('.'):
        if hasattr(obj, layer):
            obj = getattr(obj, layer)
        else:
            return None
    return obj

@compatibility(is_backward_compatible=False)
def setattr_recursive(obj, attr, value):
    if False:
        print('Hello World!')
    if '.' not in attr:
        setattr(obj, attr, value)
    else:
        layer = attr.split('.')
        setattr_recursive(getattr(obj, layer[0]), '.'.join(layer[1:]), value)

@compatibility(is_backward_compatible=False)
@dataclass
class Component:
    """
    A component serves as a container for a subgraph we want to create afterwards.
    """
    graph: torch.fx.Graph
    order: int
    name: str
    input_placeholders: List = field(default_factory=list)
    orig_inputs: List = field(default_factory=list)
    orig_outputs: List = field(default_factory=list)
    getattr_maps: Dict[torch.fx.Node, torch.fx.Node] = field(default_factory=dict)
    constructor_args: List[str] = field(default_factory=list)
    gm: Optional[torch.fx.GraphModule] = None

@compatibility(is_backward_compatible=False)
def split_by_tags(gm: torch.fx.GraphModule, tags: List[str], return_fqn_mapping: bool=False, GraphModuleCls: Type[torch.fx.GraphModule]=torch.fx.GraphModule) -> Union[torch.fx.GraphModule, Tuple[torch.fx.GraphModule, Dict[str, str]]]:
    if False:
        while True:
            i = 10
    '\n    Splits a GraphModule using tags on its graph nodes. We honor the order of\n    tags. For example, we have tags = ["a", "b", "c"], the function will create\n    the initial submodules in the order of "a", "b", "c".\n\n    To set a tag:\n    gm.graph.nodes[idx].tag = "mytag"\n\n    This will result in all nodes with the same tag being extracted and placed in their\n    own submodule. For placeholder, output and get_attr node, the tag is ignored. placeholder\n    and output nodes are created when needed while get_attr nodes get copied to submodules\n    where they are used.\n\n    Given the following module def:\n\n    class SimpleModule(torch.nn.Module):\n        def __init__(self):\n            super().__init__()\n            self.linear1 = torch.nn.Linear(...)\n            self.linear2 = torch.nn.Linear(...)\n            self.linear3 = torch.nn.Linear(...)\n\n        def forward(self, in1, in2):\n            r1 = self.linear1(in1)\n            r2 = self.linear2(in2)\n            r3 = torch.cat([r1, r2])\n            return self.linear3(r3)\n\n    Marking the node corresponding to in1 with the tag sc.REQUEST_ONLY.lower() results in the following split:\n\n    ro:\n    def forward(self, in1):\n        self = self.root\n        linear1 = self.linear1(in1)\n        return linear1\n\n    main:\n    def forward(self, in2, linear1):\n        self = self.root\n        linear2 = self.linear2(in2)\n        cat_1 = torch.cat([linear1, linear2])\n        linear3 = self.linear3(cat_1)\n        return linear3\n\n    main:\n    def forward(self, in1, in2):\n        self = self.root\n        ro_0 = self.ro_0(in1)\n        main_1 = self.main_1(in2, ro_0)\n        return main_1\n\n    Returns:\n        split_gm: torch fx graph after split\n        orig_to_split_fqn_mapping: a map between the original fqn and the fqn\n            after split for call_module and get_attr.\n    '

    def flatten(x: torch.fx.node.Argument) -> NodeList:
        if False:
            print('Hello World!')
        '\n        Stores nodes in x to a list and returns the list.\n        '
        r: NodeList = []
        map_arg(x, r.append)
        return r
    node_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    node_to_component: Dict[torch.fx.Node, Component] = {}
    tag_to_component: Dict[str, Component] = {}
    all_components: List[Component] = []
    used_in_main: Dict[torch.fx.Node, None] = {}
    main_g = torch.fx.Graph()
    main_remapping: Dict[torch.fx.Node, torch.fx.Node] = {}
    output_node: Optional[torch.fx.Node] = None
    for tag in tags:
        comp = Component(torch.fx.Graph(), len(all_components), f'{tag}')
        all_components.append(comp)
        tag_to_component[tag] = comp
    for node in gm.graph.nodes:
        if node.op == 'output':
            if output_node is not None:
                raise RuntimeError('Multiple output nodes in graph!')
            output_node = node
            continue
        if node.op == 'placeholder':
            main_remapping[node] = main_g.placeholder(node.name, type_expr=node.type)
            main_remapping[node].meta = copy.copy(node.meta)
            continue
        if node.op == 'get_attr':
            continue
        assert hasattr(node, 'tag')
        upstream_components = [node_to_component[x] for x in flatten(node.args) + flatten(node.kwargs) if x.op not in {'placeholder', 'get_attr'}]
        comp = tag_to_component[node.tag]
        node_to_component[node] = comp
        mx = max((c.order for c in upstream_components), default=0)
        assert comp.order >= mx

        def remap_func(x):
            if False:
                print('Hello World!')
            if x.op == 'get_attr':
                if x not in comp.getattr_maps:
                    comp.getattr_maps[x] = comp.graph.get_attr(x.target, type_expr=x.type)
                return comp.getattr_maps[x]
            if x.op != 'placeholder' and node_to_component[x] == comp:
                return node_remapping[x]
            if x not in comp.orig_inputs:
                comp.orig_inputs.append(x)
                placeholder = comp.graph.placeholder(x.name, type_expr=x.type)
                placeholder.meta = copy.copy(x.meta)
                comp.input_placeholders.append(placeholder)
                used_in_main[x] = None
            return comp.input_placeholders[comp.orig_inputs.index(x)]
        n = comp.graph.node_copy(node, remap_func)
        n.tag = node.tag
        node_remapping[node] = n
        node_to_component[n] = comp
    if output_node is None:
        raise RuntimeError('Graph had no output node!')
    for x in flatten(output_node.args[0]):
        if x.op == 'get_attr':
            main_remapping[x] = main_g.get_attr(x.name, type_expr=x.type)
        else:
            used_in_main[x] = None
    for n in used_in_main:
        if n.op != 'placeholder':
            node_to_component[n].orig_outputs.append(n)
    orig_to_split_fqn_mapping: Dict[str, str] = {}
    for comp in all_components:
        outs = tuple(map(node_remapping.__getitem__, comp.orig_outputs))
        comp.graph.output(outs[0] if len(outs) == 1 else outs)
        (comp.gm, comp_orig_to_split_fqn_mapping) = lift_subgraph_as_module(gm, subgraph=comp.graph, comp_name=comp.name)
        orig_to_split_fqn_mapping.update(comp_orig_to_split_fqn_mapping)
        main_node = main_g.call_module(comp.name, args=tuple(map(main_remapping.__getitem__, comp.orig_inputs)), kwargs=None)
        if len(outs) == 1:
            main_remapping[comp.orig_outputs[0]] = main_node
        else:
            for (i, o) in enumerate(comp.orig_outputs):
                main_remapping[o] = torch.fx.Proxy(main_node)[i].node
    main_g.output(map_arg(output_node.args[0], main_remapping.__getitem__))
    main_root = HolderModule({comp.name: comp.gm for comp in all_components})
    for x in flatten(output_node.args[0]):
        if x.op == 'get_attr':
            setattr(main_root, x.name, getattr_recursive(gm, x.target))
    result_gm = GraphModuleCls(main_root, main_g)
    if return_fqn_mapping:
        return (result_gm, orig_to_split_fqn_mapping)
    return result_gm