import torch
from torch.fx.node import Node
from torch.fx._symbolic_trace import symbolic_trace
from torch.fx.passes.tools_common import legalize_graph
import itertools
import operator
from typing import Dict, List, Tuple

def split_result_tensors(result: torch.Tensor, inputs: List[torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    if False:
        print('Hello World!')
    '\n    A free function for use in the merge_matmul graph transformation below that\n    splits the output from a merged matmul into the individual results for each\n    input tensor.\n\n    Arguments:\n        result: The merged matmul result tensor.\n        inputs: The list of inputs that were merged into one for the matmul.\n\n    Returns:\n        List of matmul results for each input tensor.\n    '
    if isinstance(result, torch.fx.Proxy):
        splits = [0] * len(inputs)
    else:
        splits = [x.shape[0] for x in inputs]
    return torch.split(result, splits)

def may_depend_on(a: Node, b: Node, search_depth: int=6):
    if False:
        i = 10
        return i + 15
    '\n    Determine if one node depends on another in a torch.fx.Graph.\n\n    Arguments:\n        a: The node that may have a dependency on b.\n        b: The node that a may have a dependency on.\n        search_depth: In the case of an indirect dependency, this function\n                        searches upto this many nodes away in search of a\n                        data dependency. If none is found, the function\n                        makes the conservative assumption that there is a\n                        dependency.\n\n    Returns:\n        True if a may depend on b, False if it definitely does not.\n    '
    if a == b:
        return True
    if len(a.all_input_nodes) == 0:
        return False
    if search_depth == 0:
        return True
    for inp in a.all_input_nodes:
        if may_depend_on(inp, b, search_depth - 1):
            return True
    return False

def are_nodes_independent(nodes: List[Node]):
    if False:
        print('Hello World!')
    '\n    Check if all of the given nodes are pairwise-data independent.\n\n    Arguments:\n        nodes: The nodes to check for data dependencies.\n\n    Returns:\n        True if any pair in nodes has a data dependency.\n    '
    for (i, j) in itertools.combinations(nodes, 2):
        if may_depend_on(i, j) or may_depend_on(j, i):
            return False
    return True

def merge_matmul(in_mod: torch.nn.Module):
    if False:
        for i in range(10):
            print('nop')
    '\n    A graph transformation that merges matrix multiplication operations that share the same right-hand\n    side operand into one large matrix multiplication.\n               ____      _________        _________\n      ----    |    |    |         |     M|  A * C  |\n    M| A  |  T| B  | * K|    C    | =    |---------|\n      ---- ,  |    |    |         |     T|  B * C  |\n       K       ----      ---------        ---------\n                K            R                R\n    '
    gm = symbolic_trace(in_mod)
    rhs_users: Dict[Node, List[Node]] = {}
    lhs_users: Dict[Node, List[Node]] = {}
    for node in gm.graph.nodes:
        if node.op != 'call_function' or node.target is not torch.matmul:
            continue
        (lhs, rhs) = node.args
        lhs = lhs.target if lhs.op == 'get_attr' else lhs
        rhs = rhs.target if rhs.op == 'get_attr' else rhs
        lhs_users.setdefault(lhs, []).append(node)
        rhs_users.setdefault(rhs, []).append(node)
    for (rhs, mms) in rhs_users.items():
        if len(mms) < 2:
            continue
        if not are_nodes_independent(mms):
            continue
        lhs_vals = [mm.args[0] for mm in mms]
        lhs = [gm.graph.get_attr(l) if isinstance(l, str) else l for l in lhs_vals]
        rhs = gm.graph.get_attr(rhs) if isinstance(rhs, str) else rhs
        merge_mm_cat = gm.graph.call_function(torch.cat, (lhs,), {})
        merge_mm = gm.graph.call_function(torch.matmul, (merge_mm_cat, rhs), {})
        merge_mm_split = gm.graph.call_function(split_result_tensors, (merge_mm, lhs), {})
        merge_mm_res = [gm.graph.call_function(operator.getitem, (merge_mm_split, out), {}) for out in range(len(lhs))]
        for (old, new) in zip(mms, merge_mm_res):
            old.replace_all_uses_with(new)
            gm.graph.erase_node(old)
        legalize_graph(gm)
    gm.recompile()
    gm.graph.lint()
    return gm