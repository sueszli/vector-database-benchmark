from typing import *

def strongly_connected_components(vertices: AbstractSet[str], edges: Dict[str, AbstractSet[str]]) -> Iterator[AbstractSet[str]]:
    if False:
        while True:
            i = 10
    'Compute Strongly Connected Components of a directed graph.\n\n    Args:\n      vertices: the labels for the vertices\n      edges: for each vertex, gives the target vertices of its outgoing edges\n\n    Returns:\n      An iterator yielding strongly connected components, each\n      represented as a set of vertices.  Each input vertex will occur\n      exactly once; vertices not part of a SCC are returned as\n      singleton sets.\n\n    From http://code.activestate.com/recipes/578507/.\n    '
    identified: Set[str] = set()
    stack: List[str] = []
    index: Dict[str, int] = {}
    boundaries: List[int] = []

    def dfs(v: str) -> Iterator[Set[str]]:
        if False:
            while True:
                i = 10
        index[v] = len(stack)
        stack.append(v)
        boundaries.append(index[v])
        for w in edges[v]:
            if w not in index:
                yield from dfs(w)
            elif w not in identified:
                while index[w] < boundaries[-1]:
                    boundaries.pop()
        if boundaries[-1] == index[v]:
            boundaries.pop()
            scc = set(stack[index[v]:])
            del stack[index[v]:]
            identified.update(scc)
            yield scc
    for v in vertices:
        if v not in index:
            yield from dfs(v)

def topsort(data: Dict[AbstractSet[str], Set[AbstractSet[str]]]) -> Iterable[AbstractSet[AbstractSet[str]]]:
    if False:
        print('Hello World!')
    "Topological sort.\n\n    Args:\n      data: A map from SCCs (represented as frozen sets of strings) to\n            sets of SCCs, its dependencies.  NOTE: This data structure\n            is modified in place -- for normalization purposes,\n            self-dependencies are removed and entries representing\n            orphans are added.\n\n    Returns:\n      An iterator yielding sets of SCCs that have an equivalent\n      ordering.  NOTE: The algorithm doesn't care about the internal\n      structure of SCCs.\n\n    Example:\n      Suppose the input has the following structure:\n\n        {A: {B, C}, B: {D}, C: {D}}\n\n      This is normalized to:\n\n        {A: {B, C}, B: {D}, C: {D}, D: {}}\n\n      The algorithm will yield the following values:\n\n        {D}\n        {B, C}\n        {A}\n\n    From http://code.activestate.com/recipes/577413/.\n    "
    for (k, v) in data.items():
        v.discard(k)
    for item in set.union(*data.values()) - set(data.keys()):
        data[item] = set()
    while True:
        ready = {item for (item, dep) in data.items() if not dep}
        if not ready:
            break
        yield ready
        data = {item: dep - ready for (item, dep) in data.items() if item not in ready}
    assert not data, 'A cyclic dependency exists amongst %r' % data

def find_cycles_in_scc(graph: Dict[str, AbstractSet[str]], scc: AbstractSet[str], start: str) -> Iterable[List[str]]:
    if False:
        return 10
    "Find cycles in SCC emanating from start.\n\n    Yields lists of the form ['A', 'B', 'C', 'A'], which means there's\n    a path from A -> B -> C -> A.  The first item is always the start\n    argument, but the last item may be another element, e.g.  ['A',\n    'B', 'C', 'B'] means there's a path from A to B and there's a\n    cycle from B to C and back.\n    "
    assert start in scc, (start, scc)
    assert scc <= graph.keys(), scc - graph.keys()
    graph = {src: {dst for dst in dsts if dst in scc} for (src, dsts) in graph.items() if src in scc}
    assert start in graph

    def dfs(node: str, path: List[str]) -> Iterator[List[str]]:
        if False:
            while True:
                i = 10
        if node in path:
            yield (path + [node])
            return
        path = path + [node]
        for child in graph[node]:
            yield from dfs(child, path)
    yield from dfs(start, [])