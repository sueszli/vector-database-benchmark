from collections import deque
from typing import List, Set

class DiGraph:
    """Really simple unweighted directed graph data structure to track dependencies.

    The API is pretty much the same as networkx so if you add something just
    copy their API.
    """

    def __init__(self):
        if False:
            print('Hello World!')
        self._node = {}
        self._succ = {}
        self._pred = {}
        self._node_order = {}
        self._insertion_idx = 0

    def add_node(self, n, **kwargs):
        if False:
            return 10
        'Add a node to the graph.\n\n        Args:\n            n: the node. Can we any object that is a valid dict key.\n            **kwargs: any attributes you want to attach to the node.\n        '
        if n not in self._node:
            self._node[n] = kwargs
            self._succ[n] = {}
            self._pred[n] = {}
            self._node_order[n] = self._insertion_idx
            self._insertion_idx += 1
        else:
            self._node[n].update(kwargs)

    def add_edge(self, u, v):
        if False:
            for i in range(10):
                print('nop')
        'Add an edge to graph between nodes ``u`` and ``v``\n\n        ``u`` and ``v`` will be created if they do not already exist.\n        '
        self.add_node(u)
        self.add_node(v)
        self._succ[u][v] = True
        self._pred[v][u] = True

    def successors(self, n):
        if False:
            for i in range(10):
                print('nop')
        'Returns an iterator over successor nodes of n.'
        try:
            return iter(self._succ[n])
        except KeyError as e:
            raise ValueError(f'The node {n} is not in the digraph.') from e

    def predecessors(self, n):
        if False:
            while True:
                i = 10
        'Returns an iterator over predecessors nodes of n.'
        try:
            return iter(self._pred[n])
        except KeyError as e:
            raise ValueError(f'The node {n} is not in the digraph.') from e

    @property
    def edges(self):
        if False:
            for i in range(10):
                print('nop')
        'Returns an iterator over all edges (u, v) in the graph'
        for (n, successors) in self._succ.items():
            for succ in successors:
                yield (n, succ)

    @property
    def nodes(self):
        if False:
            return 10
        'Returns a dictionary of all nodes to their attributes.'
        return self._node

    def __iter__(self):
        if False:
            print('Hello World!')
        'Iterate over the nodes.'
        return iter(self._node)

    def __contains__(self, n):
        if False:
            i = 10
            return i + 15
        'Returns True if ``n`` is a node in the graph, False otherwise.'
        try:
            return n in self._node
        except TypeError:
            return False

    def forward_transitive_closure(self, src: str) -> Set[str]:
        if False:
            for i in range(10):
                print('nop')
        'Returns a set of nodes that are reachable from src'
        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.successors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def backward_transitive_closure(self, src: str) -> Set[str]:
        if False:
            return 10
        'Returns a set of nodes that are reachable from src in reverse direction'
        result = set(src)
        working_set = deque(src)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n not in result:
                    result.add(n)
                    working_set.append(n)
        return result

    def all_paths(self, src: str, dst: str):
        if False:
            print('Hello World!')
        'Returns a subgraph rooted at src that shows all the paths to dst.'
        result_graph = DiGraph()
        forward_reachable_from_src = self.forward_transitive_closure(src)
        if dst not in forward_reachable_from_src:
            return result_graph
        working_set = deque(dst)
        while len(working_set) > 0:
            cur = working_set.popleft()
            for n in self.predecessors(cur):
                if n in forward_reachable_from_src:
                    result_graph.add_edge(n, cur)
                    working_set.append(n)
        return result_graph.to_dot()

    def first_path(self, dst: str) -> List[str]:
        if False:
            i = 10
            return i + 15
        'Returns a list of nodes that show the first path that resulted in dst being added to the graph.'
        path = []
        while dst:
            path.append(dst)
            candidates = self._pred[dst].keys()
            (dst, min_idx) = ('', None)
            for candidate in candidates:
                idx = self._node_order.get(candidate, None)
                if idx is None:
                    break
                if min_idx is None or idx < min_idx:
                    min_idx = idx
                    dst = candidate
        return list(reversed(path))

    def to_dot(self) -> str:
        if False:
            print('Hello World!')
        'Returns the dot representation of the graph.\n\n        Returns:\n            A dot representation of the graph.\n        '
        edges = '\n'.join((f'"{f}" -> "{t}";' for (f, t) in self.edges))
        return f'digraph G {{\nrankdir = LR;\nnode [shape=box];\n{edges}\n}}\n'