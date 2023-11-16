import heapq

class Solution(object):

    def minimumDistance(self, n, edges, s, marked):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type s: int\n        :type marked: List[int]\n        :rtype: int\n        '

        def dijkstra(start):
            if False:
                i = 10
                return i + 15
            best = [float('inf')] * len(adj)
            best[start] = 0
            min_heap = [(0, start)]
            while min_heap:
                (curr, u) = heapq.heappop(min_heap)
                if curr > best[u]:
                    continue
                if u in target:
                    return curr
                for (v, w) in adj[u]:
                    if curr + w >= best[v]:
                        continue
                    best[v] = curr + w
                    heapq.heappush(min_heap, (best[v], v))
            return -1
        target = set(marked)
        adj = [[] for _ in xrange(n)]
        for (u, v, w) in edges:
            adj[u].append((v, w))
        return dijkstra(s)