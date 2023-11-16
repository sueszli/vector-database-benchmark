import heapq

class Solution(object):

    def countRestrictedPaths(self, n, edges):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        adj = [[] for _ in xrange(n)]
        for (u, v, w) in edges:
            adj[u - 1].append((v - 1, w))
            adj[v - 1].append((u - 1, w))
        dist = [float('inf')] * n
        dp = [0] * n
        dist[n - 1] = 0
        dp[n - 1] = 1
        min_heap = [(0, n - 1)]
        while min_heap:
            (w, u) = heapq.heappop(min_heap)
            if w > dist[u]:
                continue
            for (v, d) in adj[u]:
                if w + d < dist[v]:
                    dist[v] = w + d
                    heapq.heappush(min_heap, (dist[v], v))
                elif w > dist[v]:
                    dp[u] = (dp[u] + dp[v]) % MOD
            if u == 0:
                break
        return dp[0]