"""Topological sorting algorithms."""
from __future__ import annotations
from typing import Any
from typing import Collection
from typing import DefaultDict
from typing import Iterable
from typing import Iterator
from typing import Sequence
from typing import Set
from typing import Tuple
from typing import TypeVar
from .. import util
from ..exc import CircularDependencyError
_T = TypeVar('_T', bound=Any)
__all__ = ['sort', 'sort_as_subsets', 'find_cycles']

def sort_as_subsets(tuples: Collection[Tuple[_T, _T]], allitems: Collection[_T]) -> Iterator[Sequence[_T]]:
    if False:
        for i in range(10):
            print('nop')
    edges: DefaultDict[_T, Set[_T]] = util.defaultdict(set)
    for (parent, child) in tuples:
        edges[child].add(parent)
    todo = list(allitems)
    todo_set = set(allitems)
    while todo_set:
        output = []
        for node in todo:
            if todo_set.isdisjoint(edges[node]):
                output.append(node)
        if not output:
            raise CircularDependencyError('Circular dependency detected.', find_cycles(tuples, allitems), _gen_edges(edges))
        todo_set.difference_update(output)
        todo = [t for t in todo if t in todo_set]
        yield output

def sort(tuples: Collection[Tuple[_T, _T]], allitems: Collection[_T], deterministic_order: bool=True) -> Iterator[_T]:
    if False:
        while True:
            i = 10
    'sort the given list of items by dependency.\n\n    \'tuples\' is a list of tuples representing a partial ordering.\n\n    deterministic_order is no longer used, the order is now always\n    deterministic given the order of "allitems".    the flag is there\n    for backwards compatibility with Alembic.\n\n    '
    for set_ in sort_as_subsets(tuples, allitems):
        yield from set_

def find_cycles(tuples: Iterable[Tuple[_T, _T]], allitems: Iterable[_T]) -> Set[_T]:
    if False:
        i = 10
        return i + 15
    edges: DefaultDict[_T, Set[_T]] = util.defaultdict(set)
    for (parent, child) in tuples:
        edges[parent].add(child)
    nodes_to_test = set(edges)
    output = set()
    for node in nodes_to_test:
        stack = [node]
        todo = nodes_to_test.difference(stack)
        while stack:
            top = stack[-1]
            for node in edges[top]:
                if node in stack:
                    cyc = stack[stack.index(node):]
                    todo.difference_update(cyc)
                    output.update(cyc)
                if node in todo:
                    stack.append(node)
                    todo.remove(node)
                    break
            else:
                node = stack.pop()
    return output

def _gen_edges(edges: DefaultDict[_T, Set[_T]]) -> Set[Tuple[_T, _T]]:
    if False:
        while True:
            i = 10
    return {(right, left) for left in edges for right in edges[left]}