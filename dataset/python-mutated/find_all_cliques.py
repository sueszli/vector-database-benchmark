"""
Finds all cliques in an undirected graph. A clique is a set of vertices in the
graph such that the subgraph is fully connected (ie. for any pair of nodes in
the subgraph there is an edge between them).
"""

def find_all_cliques(edges):
    if False:
        print('Hello World!')
    '\n    takes dict of sets\n    each key is a vertex\n    value is set of all edges connected to vertex\n    returns list of lists (each sub list is a maximal clique)\n    implementation of the basic algorithm described in:\n    Bron, Coen; Kerbosch, Joep (1973), "Algorithm 457: finding all cliques of an undirected graph",\n    '

    def expand_clique(candidates, nays):
        if False:
            print('Hello World!')
        nonlocal compsub
        if not candidates and (not nays):
            nonlocal solutions
            solutions.append(compsub.copy())
        else:
            for selected in candidates.copy():
                candidates.remove(selected)
                candidates_temp = get_connected(selected, candidates)
                nays_temp = get_connected(selected, nays)
                compsub.append(selected)
                expand_clique(candidates_temp, nays_temp)
                nays.add(compsub.pop())

    def get_connected(vertex, old_set):
        if False:
            i = 10
            return i + 15
        new_set = set()
        for neighbor in edges[str(vertex)]:
            if neighbor in old_set:
                new_set.add(neighbor)
        return new_set
    compsub = []
    solutions = []
    possibles = set(edges.keys())
    expand_clique(possibles, set())
    return solutions