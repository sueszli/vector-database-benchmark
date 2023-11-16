import collections
import heapq

class Solution(object):

    def findCheapestPrice(self, n, flights, src, dst, K):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type flights: List[List[int]]\n        :type src: int\n        :type dst: int\n        :type K: int\n        :rtype: int\n        '
        adj = collections.defaultdict(list)
        for (u, v, w) in flights:
            adj[u].append((v, w))
        best = collections.defaultdict(lambda : collections.defaultdict(lambda : float('inf')))
        best[src][K + 1] = 0
        min_heap = [(0, src, K + 1)]
        while min_heap:
            (result, u, k) = heapq.heappop(min_heap)
            if k < 0 or best[u][k] < result:
                continue
            if u == dst:
                return result
            for (v, w) in adj[u]:
                if result + w < best[v][k - 1]:
                    best[v][k - 1] = result + w
                    heapq.heappush(min_heap, (result + w, v, k - 1))
        return -1