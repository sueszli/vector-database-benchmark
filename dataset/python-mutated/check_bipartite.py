"""
Bipartite graph is a graph whose vertices can be divided into two disjoint and independent sets.
(https://en.wikipedia.org/wiki/Bipartite_graph)
"""

def check_bipartite(adj_list):
    if False:
        return 10
    '\n    Determine if the given graph is bipartite.\n\n    Time complexity is O(|E|)\n    Space complexity is O(|V|)\n    '
    vertices = len(adj_list)
    set_type = [-1 for v in range(vertices)]
    set_type[0] = 0
    queue = [0]
    while queue:
        current = queue.pop(0)
        if adj_list[current][current]:
            return False
        for adjacent in range(vertices):
            if adj_list[current][adjacent]:
                if set_type[adjacent] == set_type[current]:
                    return False
                if set_type[adjacent] == -1:
                    set_type[adjacent] = 1 - set_type[current]
                    queue.append(adjacent)
    return True