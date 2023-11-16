import heapq

class Graph(object):

    def __init__(self, n, edges):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        '
        self.__adj = [[] for _ in xrange(n)]
        for edge in edges:
            self.addEdge(edge)

    def addEdge(self, edge):
        if False:
            return 10
        '\n        :type edge: List[int]\n        :rtype: None\n        '
        (u, v, w) = edge
        self.__adj[u].append((v, w))

    def shortestPath(self, node1, node2):
        if False:
            i = 10
            return i + 15
        '\n        :type node1: int\n        :type node2: int\n        :rtype: int\n        '

        def dijkstra(adj, start, target):
            if False:
                return 10
            best = [float('inf')] * len(adj)
            best[start] = 0
            min_heap = [(best[start], start)]
            while min_heap:
                (curr, u) = heapq.heappop(min_heap)
                if curr > best[u]:
                    continue
                for (v, w) in adj[u]:
                    if not curr + w < best[v]:
                        continue
                    best[v] = curr + w
                    heapq.heappush(min_heap, (best[v], v))
            return best[target] if best[target] != float('inf') else -1
        return dijkstra(self.__adj, node1, node2)