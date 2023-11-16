from dataclasses import dataclass, field
from torch.fx.graph import Graph
from torch.fx.node import Node
from torch.fx._compatibility import compatibility
from typing import Dict, List, Any, Type, Optional, Callable
import logging
import os
__all__ = ['get_source_partitions', 'check_subgraphs_connected', 'SourcePartition']

def _init_logger():
    if False:
        print('Hello World!')
    logger = logging.getLogger(__name__)
    level = os.environ.get('PYTORCH_MATCHER_LOGLEVEL', 'WARNING').upper()
    logger.setLevel(level)
    console = logging.StreamHandler()
    formatter = logging.Formatter('%(filename)s > %(message)s')
    console.setFormatter(formatter)
    console.setLevel(level)
    logger.addHandler(console)
    logger.propagate = False
    return logger
logger = _init_logger()

@compatibility(is_backward_compatible=False)
@dataclass
class SourcePartition:
    nodes: List[Node]
    source: Any
    input_nodes: List[Node] = field(default_factory=list)
    output_nodes: List[Node] = field(default_factory=list)
    params: List[Node] = field(default_factory=list)

@compatibility(is_backward_compatible=False)
def get_source_partitions(graph: Graph, wanted_sources: List[Any], filter_fn: Optional[Callable[[Node], bool]]=None) -> Dict[Any, List[SourcePartition]]:
    if False:
        print('Hello World!')
    '\n    Args:\n        graph: The graph we want to partition\n        wanted_sources: List of sources of nodes that were decomposed from this\n            source. This can be a function (ex. torch.nn.functional.linear) or a\n            leaf module type (ex. torch.nn.Linear).\n\n    Returns:\n        Dictionary mapping sources that were given to a list of SourcePartitions\n        that correspond to the list of nodes that were decomposed from the given\n        source.\n    '
    modules: Dict[Type, Dict[str, List[Node]]] = {}
    for node in graph.nodes:
        if (source_fn_st := node.meta.get('source_fn_stack', None)) is None:
            continue
        source_fn = source_fn_st[-1]
        if source_fn[1] not in wanted_sources:
            continue
        diff_modules = modules.setdefault(source_fn[1], {})
        partition = diff_modules.setdefault(source_fn[0], [])
        partition.append(node)

    def make_partition(nodes: List[Node], module_type: Type) -> SourcePartition:
        if False:
            for i in range(10):
                print('nop')
        input_nodes = set()
        output_nodes = set()
        params = set()
        for node in nodes:
            for arg in node.args:
                if isinstance(arg, Node) and arg not in nodes:
                    input_nodes.add(arg)
            if node.op == 'get_attr':
                params.add(node)
            for user in node.users.keys():
                if user not in nodes:
                    output_nodes.add(node)
        return SourcePartition(nodes, module_type, list(input_nodes), list(output_nodes), list(params))
    ret: Dict[Type[Any], List[SourcePartition]] = {}
    if filter_fn:
        filtered_modules = {}
        for (tp, name_to_partition) in modules.items():
            filtered_name_to_partition = {name: partition for (name, partition) in name_to_partition.items() if all(map(filter_fn, partition))}
            filtered_modules[tp] = filtered_name_to_partition
        modules = filtered_modules
    for (k, v) in modules.items():
        ret[k] = [make_partition(partition, k) for partition in v.values()]
    return ret

@compatibility(is_backward_compatible=False)
def check_subgraphs_connected(subgraph1: SourcePartition, subgraph2: SourcePartition) -> bool:
    if False:
        print('Hello World!')
    '\n    Given two subgraphs A and B (in the form of a list of nodes), checks if\n    A has nodes connecting to at least one node in B -- aka there exists a node\n    in B that uses a node in A (not the other way around).\n    '
    for node in reversed(subgraph1.nodes):
        for user in node.users.keys():
            if user in subgraph2.nodes:
                return True
    return False