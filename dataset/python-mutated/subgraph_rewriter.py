from .graph_module import GraphModule
from .graph import Graph
from .node import Node
from ._symbolic_trace import symbolic_trace
from ._compatibility import compatibility
import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Set, Union
import torch
__all__ = ['Match', 'replace_pattern', 'replace_pattern_with_filters', 'ReplacedPatterns']

@compatibility(is_backward_compatible=True)
class Match(NamedTuple):
    anchor: Node
    nodes_map: Dict[Node, Node]

@compatibility(is_backward_compatible=False)
@dataclass
class ReplacedPatterns:
    anchor: Node
    nodes_map: Dict[Node, Node]
    replacements: List[Node]

def _replace_attributes(gm: GraphModule, replacement: torch.nn.Module) -> None:
    if False:
        return 10
    gm.delete_all_unused_submodules()
    if isinstance(replacement, GraphModule):
        replacement.graph.lint()

    def try_get_attr(gm: torch.nn.Module, target: str) -> Optional[Any]:
        if False:
            while True:
                i = 10
        (module_path, _, attr_name) = target.rpartition('.')
        mod: torch.nn.Module = gm.get_submodule(module_path)
        attr = getattr(mod, attr_name, None)
        return attr
    for node in gm.graph.nodes:
        if node.op == 'call_module' or node.op == 'get_attr':
            gm_attr = try_get_attr(gm, node.target)
            replacement_attr = try_get_attr(replacement, node.target)
            if gm_attr is not None:
                continue
            elif replacement_attr is not None:
                new_attr = copy.deepcopy(replacement_attr)
                if isinstance(replacement_attr, torch.nn.Module):
                    gm.add_submodule(node.target, new_attr)
                else:
                    setattr(gm, node.target, new_attr)
            else:
                raise RuntimeError('Attempted to create a "', node.op, f'" node during subgraph rewriting with target {node.target}, but the referenced attribute does not exist in the replacement GraphModule')
    gm.graph.lint()

@compatibility(is_backward_compatible=True)
def replace_pattern(gm: GraphModule, pattern: Union[Callable, GraphModule], replacement: Union[Callable, GraphModule]) -> List[Match]:
    if False:
        i = 10
        return i + 15
    '\n    Matches all possible non-overlapping sets of operators and their\n    data dependencies (``pattern``) in the Graph of a GraphModule\n    (``gm``), then replaces each of these matched subgraphs with another\n    subgraph (``replacement``).\n\n    Args:\n        ``gm``: The GraphModule that wraps the Graph to operate on\n        ``pattern``: The subgraph to match in ``gm`` for replacement\n        ``replacement``: The subgraph to replace ``pattern`` with\n\n    Returns:\n        List[Match]: A list of ``Match`` objects representing the places\n        in the original graph that ``pattern`` was matched to. The list\n        is empty if there are no matches. ``Match`` is defined as:\n\n        .. code-block:: python\n\n            class Match(NamedTuple):\n                # Node from which the match was found\n                anchor: Node\n                # Maps nodes in the pattern subgraph to nodes in the larger graph\n                nodes_map: Dict[Node, Node]\n\n    Examples:\n\n    .. code-block:: python\n\n        import torch\n        from torch.fx import symbolic_trace, subgraph_rewriter\n\n        class M(torch.nn.Module):\n            def __init__(self):\n                super().__init__()\n\n            def forward(self, x, w1, w2):\n                m1 = torch.cat([w1, w2]).sum()\n                m2 = torch.cat([w1, w2]).sum()\n                return x + torch.max(m1) + torch.max(m2)\n\n        def pattern(w1, w2):\n            return torch.cat([w1, w2]).sum()\n\n        def replacement(w1, w2):\n            return torch.stack([w1, w2])\n\n        traced_module = symbolic_trace(M())\n\n        subgraph_rewriter.replace_pattern(traced_module, pattern, replacement)\n\n    The above code will first match ``pattern`` in the ``forward``\n    method of ``traced_module``. Pattern-matching is done based on\n    use-def relationships, not node names. For example, if you had\n    ``p = torch.cat([a, b])`` in ``pattern``, you could match\n    ``m = torch.cat([a, b])`` in the original ``forward`` function,\n    despite the variable names being different (``p`` vs ``m``).\n\n    The ``return`` statement in ``pattern`` is matched based on its\n    value only; it may or may not match to the ``return`` statement in\n    the larger graph. In other words, the pattern doesn\'t have to extend\n    to the end of the larger graph.\n\n    When the pattern is matched, it will be removed from the larger\n    function and replaced by ``replacement``. If there are multiple\n    matches for ``pattern`` in the larger function, each non-overlapping\n    match will be replaced. In the case of a match overlap, the first\n    found match in the set of overlapping matches will be replaced.\n    ("First" here being defined as the first in a topological ordering\n    of the Nodes\' use-def relationships. In most cases, the first Node\n    is the parameter that appears directly after ``self``, while the\n    last Node is whatever the function returns.)\n\n    One important thing to note is that the parameters of the\n    ``pattern`` Callable must be used in the Callable itself,\n    and the parameters of the ``replacement`` Callable must match\n    the pattern. The first rule is why, in the above code block, the\n    ``forward`` function has parameters ``x, w1, w2``, but the\n    ``pattern`` function only has parameters ``w1, w2``. ``pattern``\n    doesn\'t use ``x``, so it shouldn\'t specify ``x`` as a parameter.\n    As an example of the second rule, consider replacing\n\n    .. code-block:: python\n\n        def pattern(x, y):\n            return torch.neg(x) + torch.relu(y)\n\n    with\n\n    .. code-block:: python\n\n        def replacement(x, y):\n            return torch.relu(x)\n\n    In this case, ``replacement`` needs the same number of parameters\n    as ``pattern`` (both ``x`` and ``y``), even though the parameter\n    ``y`` isn\'t used in ``replacement``.\n\n    After calling ``subgraph_rewriter.replace_pattern``, the generated\n    Python code looks like this:\n\n    .. code-block:: python\n\n        def forward(self, x, w1, w2):\n            stack_1 = torch.stack([w1, w2])\n            sum_1 = stack_1.sum()\n            stack_2 = torch.stack([w1, w2])\n            sum_2 = stack_2.sum()\n            max_1 = torch.max(sum_1)\n            add_1 = x + max_1\n            max_2 = torch.max(sum_2)\n            add_2 = add_1 + max_2\n            return add_2\n    '
    match_and_replacements = _replace_pattern(gm, pattern, replacement)
    return [Match(anchor=m.anchor, nodes_map=m.nodes_map) for m in match_and_replacements]

@compatibility(is_backward_compatible=False)
def replace_pattern_with_filters(gm: GraphModule, pattern: Union[Callable, Graph, GraphModule], replacement: Union[Callable, Graph, GraphModule], match_filters: Optional[List[Callable[['InternalMatch', Graph, Graph], bool]]]=None, ignore_literals: bool=False) -> List[ReplacedPatterns]:
    if False:
        i = 10
        return i + 15
    '\n    See replace_pattern for documentation. This function is an overload with an additional match_filter argument.\n\n    Args:\n        ``match_filters``: A list of functions that take in\n            (match: InternalMatch, original_graph: Graph, pattern_graph: Graph) and return a boolean indicating\n            whether the match satisfies the condition.\n            See matcher_utils.py for definition of InternalMatch.\n    '
    return _replace_pattern(gm, pattern, replacement, match_filters, ignore_literals)

def _replace_pattern(gm: GraphModule, pattern: Union[Callable, Graph, GraphModule], replacement: Union[Callable, Graph, GraphModule], match_filters: Optional[List[Callable[['InternalMatch', Graph, Graph], bool]]]=None, ignore_literals: bool=False) -> List[ReplacedPatterns]:
    if False:
        for i in range(10):
            print('nop')
    from torch.fx.passes.utils.matcher_utils import SubgraphMatcher, InternalMatch
    if match_filters is None:
        match_filters = []
    original_graph: Graph = gm.graph
    if isinstance(pattern, GraphModule):
        pattern_graph = pattern.graph
    elif isinstance(pattern, Graph):
        pattern_graph = pattern
    else:
        pattern_graph = symbolic_trace(pattern).graph
    if isinstance(replacement, GraphModule):
        replacement_graph = replacement.graph
    elif isinstance(replacement, Graph):
        replacement_graph = replacement
    else:
        replacement_graph = symbolic_trace(replacement).graph
    matcher = SubgraphMatcher(pattern_graph, match_output=False, match_placeholder=False, remove_overlapping_matches=True, ignore_literals=ignore_literals)
    _matches: List[InternalMatch] = matcher.match(original_graph)
    _matches = [m for m in _matches if all((match_filter(m, original_graph, pattern_graph) for match_filter in match_filters))]
    replacement_placeholders = [n for n in replacement_graph.nodes if n.op == 'placeholder']
    match_changed_node: Dict[Node, Node] = {}
    match_and_replacements = []
    for match in _matches:
        assert len(match.placeholder_nodes) == len(replacement_placeholders)
        val_map: Dict[Node, Node] = {}
        for (rn, gn) in zip(replacement_placeholders, match.placeholder_nodes):
            if isinstance(gn, Node):
                val_map[rn] = match_changed_node.get(gn, gn)
                if gn != val_map[rn]:
                    gn_ind = match.placeholder_nodes.index(gn)
                    match.placeholder_nodes[gn_ind] = match_changed_node[gn]
                    map_key = list(match.nodes_map.keys())[list(match.nodes_map.values()).index(gn)]
                    match.nodes_map[map_key] = match_changed_node[gn]
            else:
                val_map[rn] = gn
        user_nodes: Set[Node] = set()
        for n in match.returning_nodes:
            for user in n.users:
                user_nodes.add(user)
        assert user_nodes, 'The returning_nodes should have at least one user node'
        if len(user_nodes) == 1:
            first_user_node = next(iter(user_nodes))
        else:
            for n in original_graph.nodes:
                if n in user_nodes:
                    first_user_node = n
                    break
        with original_graph.inserting_before(first_user_node):
            copied_returning_nodes = original_graph.graph_copy(replacement_graph, val_map)
        if isinstance(copied_returning_nodes, Node):
            copied_returning_nodes = (copied_returning_nodes,)
        replacement_nodes: List[Node] = [v for v in val_map.values() if v not in match.placeholder_nodes]
        assert len(match.returning_nodes) == len(copied_returning_nodes)
        for (gn, copied_node) in zip(match.returning_nodes, copied_returning_nodes):
            gn.replace_all_uses_with(copied_node)
            match_changed_node[gn] = copied_node
        for node in reversed(pattern_graph.nodes):
            if node.op != 'placeholder' and node.op != 'output':
                gn = match.nodes_map[node]
                gm.graph.erase_node(gn)
        match_and_replacements.append(ReplacedPatterns(anchor=match.anchors[0], nodes_map=match.nodes_map, replacements=replacement_nodes))
    gm.recompile()
    if isinstance(replacement, torch.nn.Module):
        _replace_attributes(gm, replacement)
    return match_and_replacements