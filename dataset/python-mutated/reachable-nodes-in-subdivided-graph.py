import collections
import heapq

class Solution(object):

    def reachableNodes(self, edges, M, N):
        if False:
            while True:
                i = 10
        '\n        :type edges: List[List[int]]\n        :type M: int\n        :type N: int\n        :rtype: int\n        '
        adj = [[] for _ in xrange(N)]
        for (u, v, w) in edges:
            adj[u].append((v, w))
            adj[v].append((u, w))
        min_heap = [(0, 0)]
        best = collections.defaultdict(lambda : float('inf'))
        best[0] = 0
        count = collections.defaultdict(lambda : collections.defaultdict(int))
        result = 0
        while min_heap:
            (curr_total, u) = heapq.heappop(min_heap)
            if best[u] < curr_total:
                continue
            result += 1
            for (v, w) in adj[u]:
                count[u][v] = min(w, M - curr_total)
                next_total = curr_total + w + 1
                if next_total <= M and next_total < best[v]:
                    best[v] = next_total
                    heapq.heappush(min_heap, (next_total, v))
        for (u, v, w) in edges:
            result += min(w, count[u][v] + count[v][u])
        return result