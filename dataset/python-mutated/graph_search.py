class GraphSearch:
    """Graph search emulation in python, from source
    http://www.python.org/doc/essays/graphs/

    dfs stands for Depth First Search
    bfs stands for Breadth First Search"""

    def __init__(self, graph):
        if False:
            while True:
                i = 10
        self.graph = graph

    def find_path_dfs(self, start, end, path=None):
        if False:
            while True:
                i = 10
        path = path or []
        path.append(start)
        if start == end:
            return path
        for node in self.graph.get(start, []):
            if node not in path:
                newpath = self.find_path_dfs(node, end, path[:])
                if newpath:
                    return newpath

    def find_all_paths_dfs(self, start, end, path=None):
        if False:
            while True:
                i = 10
        path = path or []
        path.append(start)
        if start == end:
            return [path]
        paths = []
        for node in self.graph.get(start, []):
            if node not in path:
                newpaths = self.find_all_paths_dfs(node, end, path[:])
                paths.extend(newpaths)
        return paths

    def find_shortest_path_dfs(self, start, end, path=None):
        if False:
            while True:
                i = 10
        path = path or []
        path.append(start)
        if start == end:
            return path
        shortest = None
        for node in self.graph.get(start, []):
            if node not in path:
                newpath = self.find_shortest_path_dfs(node, end, path[:])
                if newpath:
                    if not shortest or len(newpath) < len(shortest):
                        shortest = newpath
        return shortest

    def find_shortest_path_bfs(self, start, end):
        if False:
            for i in range(10):
                print('nop')
        '\n        Finds the shortest path between two nodes in a graph using breadth-first search.\n\n        :param start: The node to start from.\n        :type start: str or int\n        :param end: The node to find the shortest path to.\n            :type end: str or int\n\n            :returns queue_path_to_end, dist_to[end]: A list of nodes\n        representing the shortest path from `start` to `end`, and a dictionary\n        mapping each node in the graph (except for `start`) with its distance from it\n        (in terms of hops). If no such path exists, returns an empty list and an empty\n        dictionary instead.\n        '
        queue = [start]
        dist_to = {start: 0}
        edge_to = {}
        if start == end:
            return queue
        while len(queue):
            value = queue.pop(0)
            for node in self.graph[value]:
                if node not in dist_to.keys():
                    edge_to[node] = value
                    dist_to[node] = dist_to[value] + 1
                    queue.append(node)
                    if end in edge_to.keys():
                        path = []
                        node = end
                        while dist_to[node] != 0:
                            path.insert(0, node)
                            node = edge_to[node]
                        path.insert(0, start)
                        return path

def main():
    if False:
        while True:
            i = 10
    "\n    # example of graph usage\n    >>> graph = {\n    ...     'A': ['B', 'C'],\n    ...     'B': ['C', 'D'],\n    ...     'C': ['D', 'G'],\n    ...     'D': ['C'],\n    ...     'E': ['F'],\n    ...     'F': ['C'],\n    ...     'G': ['E'],\n    ...     'H': ['C']\n    ... }\n\n    # initialization of new graph search object\n    >>> graph_search = GraphSearch(graph)\n\n    >>> print(graph_search.find_path_dfs('A', 'D'))\n    ['A', 'B', 'C', 'D']\n\n    # start the search somewhere in the middle\n    >>> print(graph_search.find_path_dfs('G', 'F'))\n    ['G', 'E', 'F']\n\n    # unreachable node\n    >>> print(graph_search.find_path_dfs('C', 'H'))\n    None\n\n    # non existing node\n    >>> print(graph_search.find_path_dfs('C', 'X'))\n    None\n\n    >>> print(graph_search.find_all_paths_dfs('A', 'D'))\n    [['A', 'B', 'C', 'D'], ['A', 'B', 'D'], ['A', 'C', 'D']]\n    >>> print(graph_search.find_shortest_path_dfs('A', 'D'))\n    ['A', 'B', 'D']\n    >>> print(graph_search.find_shortest_path_dfs('A', 'F'))\n    ['A', 'C', 'G', 'E', 'F']\n\n    >>> print(graph_search.find_shortest_path_bfs('A', 'D'))\n    ['A', 'B', 'D']\n    >>> print(graph_search.find_shortest_path_bfs('A', 'F'))\n    ['A', 'C', 'G', 'E', 'F']\n\n    # start the search somewhere in the middle\n    >>> print(graph_search.find_shortest_path_bfs('G', 'F'))\n    ['G', 'E', 'F']\n\n    # unreachable node\n    >>> print(graph_search.find_shortest_path_bfs('A', 'H'))\n    None\n\n    # non existing node\n    >>> print(graph_search.find_shortest_path_bfs('A', 'X'))\n    None\n    "
if __name__ == '__main__':
    import doctest
    doctest.testmod()