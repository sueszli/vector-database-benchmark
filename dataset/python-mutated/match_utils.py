import sys
import torch
from torch.fx.graph import Graph, Node
from torch.ao.quantization.utils import Pattern
from .quantize_handler import QuantizeHandler
from ..qconfig import QConfigAny
from ..utils import MatchAllNode
from .graph_module import _is_observed_standalone_module
from torch.nn.utils.parametrize import type_before_parametrizations
from typing import Any, Dict, List, Callable, Optional, Tuple, Type, Set, Iterable
__all__: List[str] = []
_MatchResult = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler]
_MatchResultWithQConfig = Tuple[Node, List[Node], Optional[Pattern], QuantizeHandler, QConfigAny]

def _is_match(modules, node, pattern, max_uses=sys.maxsize):
    if False:
        print('Hello World!')
    ' Matches a node in fx against a pattern\n    '
    if isinstance(pattern, tuple):
        (self_match, *arg_matches) = pattern
        if self_match is getattr:
            assert len(pattern) == 2, 'Expecting getattr pattern to have two elements'
            arg_matches = []
    else:
        self_match = pattern
        arg_matches = []
    if isinstance(self_match, type) and issubclass(self_match, MatchAllNode):
        return True
    if node == pattern:
        return True
    if not isinstance(node, Node) or len(node.users) > max_uses:
        return False
    if isinstance(self_match, type) and issubclass(self_match, torch.nn.Module):
        if node.op != 'call_module':
            return False
        if not type_before_parametrizations(modules[node.target]) == self_match:
            return False
    elif callable(self_match):
        if node.op != 'call_function' or node.target is not self_match:
            return False
        elif node.target is getattr:
            if node.args[1] != pattern[1]:
                return False
    elif isinstance(self_match, str):
        if node.op != 'call_method' or node.target != self_match:
            return False
    elif node.target != self_match:
        return False
    if not arg_matches:
        return True
    if len(arg_matches) != len(node.args):
        return False
    return all((_is_match(modules, node, arg_match, max_uses=1) for (node, arg_match) in zip(node.args, arg_matches)))

def _find_matches(graph: Graph, modules: Dict[str, torch.nn.Module], patterns: Dict[Pattern, QuantizeHandler], root_node_getter_mapping: Dict[Pattern, Callable], standalone_module_names: Optional[List[str]]=None, standalone_module_classes: Optional[List[Type]]=None, custom_module_classes: Optional[List[Any]]=None) -> Dict[str, _MatchResult]:
    if False:
        for i in range(10):
            print('nop')
    "\n    Matches the nodes in the input graph to quantization patterns, and\n    outputs the information needed to quantize them in future steps.\n\n    Inputs:\n      - graph: an fx.Graph object\n      - modules: a mapping of fully qualified module name to instance,\n          for example, {'foo': ModuleFoo, ...}\n      - patterns: a mapping from a tuple of nodes in reverse order to\n          uninitialized QuantizeHandler subclass.\n\n    Outputs a map of\n      node_name ->\n        (node, matched_values, matched_pattern, QuantizeHandler instance,\n         qconfig)\n\n    For example, {\n      'relu_1': (relu_1, [relu_1], torch.nn.functional.relu,\n                 <CopyNodeQuantizeHandler instance>, QConfig(...)),\n      ...\n    }\n    "
    if custom_module_classes is None:
        custom_module_classes = []
    if standalone_module_classes is None:
        standalone_module_classes = []
    if standalone_module_names is None:
        standalone_module_names = []
    match_map: Dict[str, _MatchResult] = {}
    all_matched: Set[str] = set()

    def _recursive_record_node_in_match_map(last_node, match_map, node_pattern, matched_node_pattern, pattern, match_value):
        if False:
            return 10
        if isinstance(node_pattern, Node):
            match_map[node_pattern.name] = (last_node, matched_node_pattern, pattern, match_value)
        elif not isinstance(node_pattern, Iterable):
            return
        else:
            for n in node_pattern:
                _recursive_record_node_in_match_map(last_node, match_map, n, matched_node_pattern, pattern, match_value)

    def record_match(pattern, node, last_node, matched_node_pattern, match_map):
        if False:
            i = 10
            return i + 15
        if isinstance(pattern, tuple):
            (s, *args) = pattern
            is_single_arg = len(args) == 1
            current_node_pattern: List[Node] = []
            record_match(s, node, last_node, matched_node_pattern, match_map)
            if pattern[0] is not getattr:
                for (subpattern, arg) in zip(args, node.args):
                    record_match(subpattern, arg, node, current_node_pattern, match_map)
            if len(current_node_pattern) > 1:
                if is_single_arg:
                    matched_node_pattern.append(tuple(current_node_pattern))
                else:
                    matched_node_pattern.extend(list(current_node_pattern))
            else:
                matched_node_pattern.append(current_node_pattern[0])
        else:
            matched_node_pattern.append(node)
    for node in reversed(graph.nodes):
        if node.name not in match_map and node.name not in all_matched:
            for (pattern, quantize_handler_cls) in patterns.items():
                root_node_getter = root_node_getter_mapping.get(pattern, None)
                if _is_match(modules, node, pattern) and node.name not in match_map:
                    matched_node_pattern: List[Node] = []
                    record_match(pattern, node, node, matched_node_pattern, match_map)
                    quantize_handler = quantize_handler_cls(matched_node_pattern, modules, root_node_getter)
                    last_node = node
                    _recursive_record_node_in_match_map(last_node, match_map, matched_node_pattern, matched_node_pattern, pattern, quantize_handler)
                    break
    assert modules is not None
    for node in graph.nodes:
        if node.op == 'call_module' and type(modules[node.target]) in custom_module_classes:
            match_map[node.name] = (node, node, None, QuantizeHandler(node, modules, is_custom_module=True))

    def is_standalone_module(node_target: str, modules: Dict[str, torch.nn.Module]):
        if False:
            return 10
        assert modules is not None
        return node_target in standalone_module_names or type(modules[node_target]) in standalone_module_classes
    for node in graph.nodes:
        if node.op == 'call_module' and (is_standalone_module(node.target, modules) or _is_observed_standalone_module(modules[node.target])):
            match_map[node.name] = (node, node, None, QuantizeHandler(node, modules, is_standalone_module=True))
    return match_map