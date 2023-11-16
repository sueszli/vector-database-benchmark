from __future__ import print_function as _
from __future__ import division as _
from __future__ import absolute_import as _
from ..basic_graph_ops import delete_node
import logging
import sys
sys.setrecursionlimit(5000)

def _all_assert_leaves(gdict, nodename, memo):
    if False:
        for i in range(10):
            print('nop')
    "\n    Does the given node lead to only assertions?\n\n    Args:\n        gdict (dict): The node's graph.\n        nodename (str): The name of the node to test.\n        memo (dict): Storage for memoization.\n    "
    work = [nodename]
    while True:
        assert len(work) <= len(gdict)
        node = gdict[work.pop()]
        if not isinstance(memo.get(node.name), bool):
            memo[node.name] = None
            outputs = node.outputs
            if len(outputs) == 0:
                memo[node.name] = node.op in ('Assert', 'CheckNumerics')
            else:
                outputs_to_process = [n for n in outputs if n not in memo]
                if len(outputs_to_process) == 0:
                    memo[node.name] = all((memo[n] for n in outputs))
                else:
                    work.append(node.name)
                    work.extend(outputs_to_process)
        if len(work) == 0:
            return memo[node.name]

def delete_asserts(tfssa):
    if False:
        while True:
            i = 10
    '\n    Delete all nodes that lead only to assertions.\n    '
    delete_count = 0
    for f in tfssa.functions.values():
        memo = {}
        for n in f.graph:
            _all_assert_leaves(f.graph, n, memo)
        for m in memo:
            if memo[m]:
                delete_count += 1
                delete_node(f.graph, m)
    logging.debug('%d assert nodes deleted', delete_count)
    return delete_count