class Solution(object):

    def findShortestCycle(self, n, edges):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '
        INF = float('inf')

        def bfs(u):
            if False:
                i = 10
                return i + 15
            result = INF
            dist = [float('inf')] * len(adj)
            dist[u] = 0
            q = [u]
            while q:
                new_q = []
                for u in q:
                    for v in adj[u]:
                        if dist[v] != INF:
                            assert abs(dist[v] - dist[u]) <= 1
                            if dist[v] != dist[u] - 1:
                                result = min(result, 1 + dist[u] + dist[v])
                            continue
                        dist[v] = dist[u] + 1
                        new_q.append(v)
                if result != INF:
                    break
                q = new_q
            return result
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        result = min((bfs(u) for u in xrange(n)))
        return result if result != INF else -1