"""
Determination of single-source shortest-path.
"""

def bellman_ford(graph, source):
    if False:
        return 10
    '\n    This Bellman-Ford Code is for determination whether we can get\n    shortest path from given graph or not for single-source shortest-paths problem.\n    In other words, if given graph has any negative-weight cycle that is reachable\n    from the source, then it will give answer False for "no solution exits".\n    For argument graph, it should be a dictionary type\n    such as\n    graph = {\n        \'a\': {\'b\': 6, \'e\': 7},\n        \'b\': {\'c\': 5, \'d\': -4, \'e\': 8},\n        \'c\': {\'b\': -2},\n        \'d\': {\'a\': 2, \'c\': 7},\n        \'e\': {\'b\': -3}\n    }\n    '
    weight = {}
    pre_node = {}
    initialize_single_source(graph, source, weight, pre_node)
    for _ in range(1, len(graph)):
        for node in graph:
            for adjacent in graph[node]:
                if weight[adjacent] > weight[node] + graph[node][adjacent]:
                    weight[adjacent] = weight[node] + graph[node][adjacent]
                    pre_node[adjacent] = node
    for node in graph:
        for adjacent in graph[node]:
            if weight[adjacent] > weight[node] + graph[node][adjacent]:
                return False
    return True

def initialize_single_source(graph, source, weight, pre_node):
    if False:
        while True:
            i = 10
    '\n    Initialize data structures for Bellman-Ford algorithm.\n    '
    for node in graph:
        weight[node] = float('inf')
        pre_node[node] = None
    weight[source] = 0