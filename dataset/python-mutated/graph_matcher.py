import collections
import enum
import torch
toq = torch.ops.quantized
from torch.fx import GraphModule
from torch.fx.graph import Graph, Node
from torch.ao.quantization.utils import getattr_from_fqn
from .ns_types import NSSubgraph, NSNodeTargetType
from .mappings import get_base_name_to_sets_of_related_ops, get_unmatchable_types_map
from .pattern_utils import get_type_a_related_to_b, get_reversed_fusions, end_node_matches_reversed_fusion
from torch.ao.quantization import ObserverBase, FakeQuantizeBase
from typing import Dict, Tuple, List, Optional, Set, Any

def _get_output_nodes(g: Graph) -> List[Node]:
    if False:
        return 10
    return [n for n in g.nodes if n.op == 'output']

class _NSGraphMatchableSubgraphsIterator:
    """
    Iterates through the graph of gm, starting with the output nodes
    and continuing backwards.
    1. Returns matchable subgraphs, in order. A subgraph is defined by
       (start_node, end_node).
    2. Skips over non-matchable subgraphs
    """

    def __init__(self, gm: GraphModule, non_matchable_functions: Set[NSNodeTargetType], non_matchable_modules: Set[NSNodeTargetType], non_matchable_methods: Set[NSNodeTargetType]):
        if False:
            for i in range(10):
                print('nop')
        self.gm: GraphModule = gm
        self.non_matchable_functions: Set[NSNodeTargetType] = non_matchable_functions
        self.non_matchable_modules: Set[NSNodeTargetType] = non_matchable_modules
        self.non_matchable_methods: Set[NSNodeTargetType] = non_matchable_methods
        self.seen_nodes: Set[Node] = set()
        self.stack: List[Node] = []
        for start_node in _get_output_nodes(self.gm.graph):
            self.stack.append(start_node)

    def __iter__(self):
        if False:
            for i in range(10):
                print('nop')
        return self

    def __next__(self) -> NSSubgraph:
        if False:
            return 10
        '\n        Returns the next matchable subgraph.\n        '
        while len(self.stack) > 0:
            cur_end_node = self.stack.pop()
            if cur_end_node in self.seen_nodes:
                continue
            cur_start_node = cur_end_node
            cur_base_op_node = cur_end_node
            for (_reverse_fusion_ops, base_op_idx) in get_reversed_fusions():
                is_match = end_node_matches_reversed_fusion(cur_end_node, _reverse_fusion_ops, self.gm, self.seen_nodes)
                if is_match:
                    for rev_fusion_idx in range(len(_reverse_fusion_ops) - 1):
                        self.seen_nodes.add(cur_start_node)
                        cur_start_node = cur_start_node.args[0]
                        rev_base_op_idx = len(_reverse_fusion_ops) - 2 - base_op_idx
                        if rev_fusion_idx == rev_base_op_idx:
                            cur_base_op_node = cur_start_node
                    break
            self.seen_nodes.add(cur_start_node)
            for arg in cur_start_node.all_input_nodes:
                self._recursively_add_node_arg_to_stack(arg)
            if not self._is_matchable(cur_base_op_node):
                continue
            if cur_end_node.op == 'call_module' and cur_start_node is cur_end_node:
                maybe_obs = getattr_from_fqn(self.gm, cur_end_node.target)
                if isinstance(maybe_obs, (ObserverBase, FakeQuantizeBase)):
                    continue
            return NSSubgraph(start_node=cur_start_node, end_node=cur_end_node, base_op_node=cur_base_op_node)
        raise StopIteration

    def _recursively_add_node_arg_to_stack(self, arg: Any) -> None:
        if False:
            for i in range(10):
                print('nop')
        '\n        Adds all of the nodes in this arg to the stack, properly navigating\n        through list, dicts and tuples.\n        '
        if isinstance(arg, Node):
            self.stack.append(arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_list) or type(arg) is tuple:
            for inner_arg in arg:
                self._recursively_add_node_arg_to_stack(inner_arg)
        elif isinstance(arg, torch.fx.immutable_collections.immutable_dict):
            for value in arg.values():
                self._recursively_add_node_arg_to_stack(value)

    def _is_matchable(self, node: Node) -> bool:
        if False:
            print('Hello World!')
        if node.op == 'call_function':
            return node.target not in self.non_matchable_functions
        elif node.op == 'call_module':
            assert isinstance(node.target, str)
            target_mod = getattr_from_fqn(self.gm, node.target)
            return not any((isinstance(target_mod, t) for t in self.non_matchable_modules))
        elif node.op == 'call_method':
            return node.target not in self.non_matchable_methods
        else:
            return False

class GraphMatchingException(Exception):
    """
    Exception raised when two graphs cannot be matched.
    """
    pass

class SubgraphTypeRelationship(enum.Enum):
    EQUAL = enum.auto()
    EQUAL_BUT_UKNOWN = enum.auto()
    RELATED_BUT_NOT_EQUAL = enum.auto()
    NOT_RELATED = enum.auto()

def _get_subgraph_relationship_type(subgraph_a: NSSubgraph, subgraph_b: NSSubgraph, gm_a: GraphModule, gm_b: GraphModule, type_a_related_to_b: Set[Tuple[NSNodeTargetType, NSNodeTargetType]]) -> SubgraphTypeRelationship:
    if False:
        return 10
    node_a = subgraph_a.base_op_node
    node_b = subgraph_b.base_op_node
    if node_a.op != node_b.op:
        if not (node_a.op in ('call_function', 'call_method') and node_b.op in ('call_function', 'call_method')):
            return SubgraphTypeRelationship.NOT_RELATED
    if node_a.op in ('call_function', 'call_method'):
        key = (node_a.target, node_b.target)
        if key not in type_a_related_to_b:
            if node_a.target == node_b.target:
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            else:
                return SubgraphTypeRelationship.NOT_RELATED
        if node_a.target == node_b.target:
            node_a_has_prev = subgraph_a.base_op_node == subgraph_a.start_node
            node_b_has_prev = subgraph_b.base_op_node == subgraph_b.start_node
            if node_a_has_prev and (not node_b_has_prev):
                return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            elif not node_a_has_prev and node_b_has_prev:
                return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
            elif not node_a_has_prev and (not node_b_has_prev):
                return SubgraphTypeRelationship.EQUAL
            else:
                return SubgraphTypeRelationship.EQUAL
        if key in type_a_related_to_b:
            return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
        else:
            return SubgraphTypeRelationship.NOT_RELATED
    elif node_a.op == 'call_module':
        assert subgraph_a.base_op_node == subgraph_a.start_node and subgraph_b.base_op_node == subgraph_b.start_node, 'Matching call_module patterns where base_op_node != start_node is not supported yet'
        assert isinstance(node_a.target, str)
        mod_a = getattr_from_fqn(gm_a, node_a.target)
        assert isinstance(node_b.target, str)
        mod_b = getattr_from_fqn(gm_b, node_b.target)
        key = (type(mod_a), type(mod_b))
        if key not in type_a_related_to_b:
            if type(mod_a) == type(mod_b):
                return SubgraphTypeRelationship.EQUAL_BUT_UKNOWN
            else:
                return SubgraphTypeRelationship.NOT_RELATED
        elif type(mod_a) == type(mod_b):
            return SubgraphTypeRelationship.EQUAL
        else:
            return SubgraphTypeRelationship.RELATED_BUT_NOT_EQUAL
    return SubgraphTypeRelationship.NOT_RELATED

def _get_name_for_subgraph(subgraph_a: NSSubgraph, gm_a: GraphModule, base_name_to_sets_of_related_ops: Dict[str, Set[NSNodeTargetType]], existing_names: Set[str]) -> str:
    if False:
        i = 10
        return i + 15
    "\n    Returns a unique name for a subgraph. This name is based on two things:\n    1. the name of the set containing the underlying type of the base op in the\n       subgraph (i.e. 'torch.nn.functional.linear' if this is related to a linear op)\n    2. the number of previous subgraphs with related underlying type of the base op\n\n    For example, in the graph\n\n    linear0 -> relu0 -> linear1 -> relu1\n\n    The subgraphs are (linear0, relu0) and (linear1, relu1).  If we iterate\n    from the output node backwards, the name given to (linear1, relu1) will be\n    `base_op_torch.nn.functional.linear_0`, and the name given to (linear0, relu0)\n    will be `base_op_torch.nn.functional.linear_1`.\n\n    Why are we not just using the node name? Answer: because of two requirements:\n    A. fusions must be supported\n    B. some Numeric Suite APIs can be called without having all of the models in memory\n\n    For example, let's say we need to match nodes of\n\n    (1) ... -> linear0 -> relu0 -> ...\n\n    And\n\n    (2) ... -> linear_relu0 -> ...\n\n    Without being able to inspect them together. With the current naming scheme, if\n    we iterate through both of these graphs in the same order, and assuming the rest\n    of the graphs match, both of these subgraphs will get the same name without\n    (1) and (2) knowing anything about each other.\n    "
    target_type = _get_node_target_type(subgraph_a.base_op_node, gm_a)
    target_base_type = None
    for (base_name, sets_of_related_ops) in base_name_to_sets_of_related_ops.items():
        if target_type in sets_of_related_ops:
            target_base_type = base_name
    target_base_name = 'base_op_' + str(target_base_type)
    counter = 0
    proposed_name = target_base_name + '_' + str(counter)
    while proposed_name in existing_names:
        counter += 1
        proposed_name = target_base_name + '_' + str(counter)
    existing_names.add(proposed_name)
    return proposed_name

def _get_node_target_type(node: Node, gm: GraphModule) -> Optional[NSNodeTargetType]:
    if False:
        print('Hello World!')
    if node.op in ('call_function', 'call_method'):
        return node.target
    elif node.op == 'call_module':
        assert isinstance(node.target, str)
        mod = getattr_from_fqn(gm, node.target)
        return type(mod)
    return None

def get_matching_subgraph_pairs(gm_a: GraphModule, gm_b: GraphModule, base_name_to_sets_of_related_ops: Optional[Dict[str, Set[NSNodeTargetType]]]=None, unmatchable_types_map: Optional[Dict[str, Set[NSNodeTargetType]]]=None) -> Dict[str, Tuple[NSSubgraph, NSSubgraph]]:
    if False:
        while True:
            i = 10
    '\n    Matches matchable subgraphs of graph_a to graph_b.\n\n    For a node, "matchable" is defined as a node which is not an observer,\n    fake_quants, quant or dequant.\n\n    A subgraph can contain one or more nodes.  A subgraph is matchable if\n    at least one node inside of it is matchable.  Currently, all nodes in\n    a subgraph must be matchable (because we assume no observers will be\n    inserted in the middle of a fusion).\n\n    A subgraph is defined by (start_node, end_node).  We assume that only\n    start_node and end_node are linked with the surrounding graph, all other\n    nodes in a subgraph are self-contained.\n\n    A pair of nodes is "related" if both nodes represent the same mathematical\n    operation across different quantization flavors. For example,\n    `F.linear` and `torch.ops.quantized.linear` are related, and\n    `F.linear` and `torch.nn.Conv` are not related.\n\n    For each matchable pair of nodes node_a and node_b, they will match\n    if node_a and node_b are related.\n\n    For graphs A and B, they will match iff:\n    1. the number of matchable subgraphs in A and B is equivalent\n    2. when iterating through the matchable subgraphs of A and B in the same order, each\n       corresponding pair of base nodes is related.\n\n    This enables us to find the corresponding subgraphs between\n    graphs of related models.  For example, if we had two graphs such as:\n\n    graph_a: x0 -> conv_0 (type: nn.Conv2d) -> obs_0 -> x1\n             w  -/\n             b  -/\n\n    graph_b: x0 -> quant_0 -> qconv_0 (type: nnq.Conv2d) -> dequant_0 -> x1\n           packed_params_0 -/\n\n    This function will return the following result:\n    {\n        \'conv_0\': (  # the name of the node in graph_b\n          (conv_0, conv_0),  # (start_node_a, end_node_a)\n          (qconv_0, qconv_0),  # (start_node_b, end_node_b)\n        ),\n    }\n\n    Or, if we have a fusion pattern,\n\n    graph_a: x0 -> linear_0 -> relu_0 -> obs_0 -> x1\n             w  -/\n             b  -/\n\n    graph_b: x0 -> quant_0 -> linear_relu_0 -> dequant_0 -> x1\n           packed_params_0 -/\n\n    This function will return the following result:\n    {\n        \'linear_relu_0\': (  # the name of the node in graph_b\n          (linear_0, relu_0),  # (start_node_a, end_node_a)\n          (linear_relu_0, linear_relu_0),  # (start_node_b, end_node_b)\n        ),\n    }\n    '
    if unmatchable_types_map is None:
        unmatchable_types_map = get_unmatchable_types_map()
    non_matchable_functions = unmatchable_types_map['funs_unmatchable']
    non_matchable_modules = unmatchable_types_map['mods_unmatchable']
    non_matchable_methods = unmatchable_types_map['meths_unmatchable']
    graph_a_iterator = _NSGraphMatchableSubgraphsIterator(gm_a, non_matchable_functions, non_matchable_modules, non_matchable_methods)
    graph_b_iterator = _NSGraphMatchableSubgraphsIterator(gm_b, non_matchable_functions, non_matchable_modules, non_matchable_methods)
    results = collections.OrderedDict()
    if base_name_to_sets_of_related_ops is None:
        base_name_to_sets_of_related_ops = get_base_name_to_sets_of_related_ops()
    type_a_related_to_b = get_type_a_related_to_b(base_name_to_sets_of_related_ops)
    existing_names_a: Set[str] = set()
    existing_names_b: Set[str] = set()
    while True:
        (cur_subgraph_a, cur_subgraph_b) = (None, None)
        try:
            cur_subgraph_a = next(graph_a_iterator)
        except StopIteration:
            pass
        try:
            cur_subgraph_b = next(graph_b_iterator)
        except StopIteration:
            pass
        (type_start_a, type_start_b) = (None, None)
        if cur_subgraph_a is not None:
            type_start_a = _get_node_target_type(cur_subgraph_a.start_node, gm_a)
        if cur_subgraph_b is not None:
            type_start_b = _get_node_target_type(cur_subgraph_b.start_node, gm_b)
        if cur_subgraph_a is not None and cur_subgraph_b is not None:
            subgraph_relationship = _get_subgraph_relationship_type(cur_subgraph_a, cur_subgraph_b, gm_a, gm_b, type_a_related_to_b)
            if subgraph_relationship == SubgraphTypeRelationship.NOT_RELATED:
                msg = f'\nThe subgraphs\n({cur_subgraph_a}, {type_start_a}) and\n({cur_subgraph_b}, {type_start_b})\nare not related. Please ensure that the two models you pass in have the same number\nof subgraphs, and each pair of subgraphs is related to each other.'
                raise GraphMatchingException(msg)
            elif subgraph_relationship == SubgraphTypeRelationship.EQUAL_BUT_UKNOWN:
                continue
            key_name_a = _get_name_for_subgraph(cur_subgraph_a, gm_a, base_name_to_sets_of_related_ops, existing_names_a)
            key_name_b = _get_name_for_subgraph(cur_subgraph_b, gm_b, base_name_to_sets_of_related_ops, existing_names_b)
            assert key_name_a == key_name_b, f'Subgraph names {key_name_a} and {key_name_b} do not match'
            results[key_name_a] = (cur_subgraph_a, cur_subgraph_b)
            continue
        elif cur_subgraph_a is None and cur_subgraph_b is None:
            break
        else:
            msg = f'\nAttempting to match\n({cur_subgraph_a}, {type_start_a}) and\n({cur_subgraph_b}, {type_start_b}),\none of which is empty. Please ensure that the two models you pass in have the same number\nof subgraphs.'
            raise GraphMatchingException(msg)
    results = collections.OrderedDict(reversed(list(results.items())))
    return results