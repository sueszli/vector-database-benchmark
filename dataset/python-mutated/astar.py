import cipheycore

class Node:
    """
    A node has a value associated with it
    Calculated from the heuristic
    """

    def __init__(self, config, h: float=None, edges: (any, float)=None, ctext: str=None):
        if False:
            while True:
                i = 10
        self.weight = h
        self.edges = edges
        self.ctext = ctext
        self.h = h
        self.path = []
        self.information_content = config.cache.get_or_update(self.text, 'cipheycore::info_content', lambda : cipheycore.info_content(self.ctext))

    def __le__(self, node2):
        if False:
            print('Hello World!')
        return self.x <= node2.x

    def __lt__(self, node2):
        if False:
            return 10
        return self.x < node2.x

    def append_edge(self, edge):
        if False:
            while True:
                i = 10
        self.edges.append(edge)

    def get_edges(self):
        if False:
            return 10
        return self.edges

class Graph:

    def __init__(self, adjacency_list):
        if False:
            for i in range(10):
                print('nop')
        '\n        adjacency list: basically the graph\n        '
        self.adjacency_list = adjacency_list
        self.original_input = cipheycore.info_content(input)

    def get_neighbors(self, v):
        if False:
            print('Hello World!')
        try:
            return self.adjacency_list[v]
        except KeyError:
            return []

    def heuristic(self, n: Node):
        if False:
            while True:
                i = 10
        return n.info_content / self.original_input

    def a_star_algorithm(self, start_node: Node, stop_node: Node):
        if False:
            return 10
        open_list = set([start_node])
        closed_list = set([])
        g = {}
        g[start_node] = 0
        parents = {}
        parents[start_node] = start_node
        while len(open_list) > 0:
            print(f'The open list is {open_list}')
            n = None
            for v in open_list:
                print(f'The for loop node v is {v}')
                if n is None or g[v] + self.h(v) < g[n] + self.h(n):
                    n = v
                    print(f'The value of n is {n}')
            if n is None:
                print('Path does not exist!')
                return None
            if n == stop_node:
                print('n is the stop node, we are stopping!')
                reconst_path = []
                while parents[n] != n:
                    reconst_path.append(n)
                    n = parents[n]
                reconst_path.append(start_node)
                reconst_path.reverse()
                print('Path found: {}'.format(reconst_path))
                return reconst_path
            print(n)
            for (m, weight) in self.get_neighbors(n):
                print(f'And the iteration is ({m}, {weight})')
                if m not in open_list and m not in closed_list:
                    open_list.add(m)
                    parents[m] = n
                    g[m] = g[n] + weight
                elif g[m] > g[n] + weight:
                    g[m] = g[n] + weight
                    parents[m] = n
                    if m in closed_list:
                        closed_list.remove(m)
                        open_list.add(m)
            open_list.remove(n)
            closed_list.add(n)
            print('\n')
        print('Path does not exist!')
        return None
adjacency_list = {'A': [('B', 1), ('C', 3), ('D', 7)], 'B': [('D', 5)], 'C': [('D', 12)]}
A = Node(1)
B = Node(7)
C = Node(9)
D = Node(16)
A.edges = [(B, 1), (C, 3), (D, 7)]
B.edges = [(D, 5)]
C.edges = [(D, 12)]
adjacency_list = {A: A.edges, B: B.edges, C: C.edges}
graph1 = Graph(adjacency_list)
graph1.a_star_algorithm(A, D)
'\nMaybe after it\n'