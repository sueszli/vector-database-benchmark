"""
altgraph.GraphUtil - Utility classes and functions
==================================================
"""
import random
from collections import deque
from altgraph import Graph, GraphError

def generate_random_graph(node_num, edge_num, self_loops=False, multi_edges=False):
    if False:
        print('Hello World!')
    '\n    Generates and returns a :py:class:`~altgraph.Graph.Graph` instance with\n    *node_num* nodes randomly connected by *edge_num* edges.\n    '
    g = Graph.Graph()
    if not multi_edges:
        if self_loops:
            max_edges = node_num * node_num
        else:
            max_edges = node_num * (node_num - 1)
        if edge_num > max_edges:
            raise GraphError("inconsistent arguments to 'generate_random_graph'")
    nodes = range(node_num)
    for node in nodes:
        g.add_node(node)
    while 1:
        head = random.choice(nodes)
        tail = random.choice(nodes)
        if head == tail and (not self_loops):
            continue
        if g.edge_by_node(head, tail) is not None and (not multi_edges):
            continue
        g.add_edge(head, tail)
        if g.number_of_edges() >= edge_num:
            break
    return g

def generate_scale_free_graph(steps, growth_num, self_loops=False, multi_edges=False):
    if False:
        for i in range(10):
            print('nop')
    '\n    Generates and returns a :py:class:`~altgraph.Graph.Graph` instance that\n    will have *steps* \\* *growth_num* nodes and a scale free (powerlaw)\n    connectivity. Starting with a fully connected graph with *growth_num*\n    nodes at every step *growth_num* nodes are added to the graph and are\n    connected to existing nodes with a probability proportional to the degree\n    of these existing nodes.\n    '
    graph = Graph.Graph()
    store = []
    for i in range(growth_num):
        for j in range(i + 1, growth_num):
            store.append(i)
            store.append(j)
            graph.add_edge(i, j)
    for node in range(growth_num, steps * growth_num):
        graph.add_node(node)
        while graph.out_degree(node) < growth_num:
            nbr = random.choice(store)
            if node == nbr and (not self_loops):
                continue
            if graph.edge_by_node(node, nbr) and (not multi_edges):
                continue
            graph.add_edge(node, nbr)
        for nbr in graph.out_nbrs(node):
            store.append(node)
            store.append(nbr)
    return graph

def filter_stack(graph, head, filters):
    if False:
        print('Hello World!')
    '\n    Perform a walk in a depth-first order starting\n    at *head*.\n\n    Returns (visited, removes, orphans).\n\n    * visited: the set of visited nodes\n    * removes: the list of nodes where the node\n      data does not all *filters*\n    * orphans: tuples of (last_good, node),\n      where node is not in removes, is directly\n      reachable from a node in *removes* and\n      *last_good* is the closest upstream node that is not\n      in *removes*.\n    '
    (visited, removes, orphans) = ({head}, set(), set())
    stack = deque([(head, head)])
    get_data = graph.node_data
    get_edges = graph.out_edges
    get_tail = graph.tail
    while stack:
        (last_good, node) = stack.pop()
        data = get_data(node)
        if data is not None:
            for filtfunc in filters:
                if not filtfunc(data):
                    removes.add(node)
                    break
            else:
                last_good = node
        for edge in get_edges(node):
            tail = get_tail(edge)
            if last_good is not node:
                orphans.add((last_good, tail))
            if tail not in visited:
                visited.add(tail)
                stack.append((last_good, tail))
    orphans = [(lg, tl) for (lg, tl) in orphans if tl not in removes]
    return (visited, removes, orphans)