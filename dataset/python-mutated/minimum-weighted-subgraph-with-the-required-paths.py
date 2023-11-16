import heapq

class Solution(object):

    def minimumWeight(self, n, edges, src1, src2, dest):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type src1: int\n        :type src2: int\n        :type dest: int\n        :rtype: int\n        '

        def dijkstra(adj, start):
            if False:
                return 10
            best = [float('inf')] * len(adj)
            best[start] = 0
            min_heap = [(0, start)]
            while min_heap:
                (curr, u) = heapq.heappop(min_heap)
                if best[u] < curr:
                    continue
                for (v, w) in adj[u]:
                    if best[v] <= curr + w:
                        continue
                    best[v] = curr + w
                    heapq.heappush(min_heap, (curr + w, v))
            return best
        (adj1, adj2) = [[[] for _ in xrange(n)] for _ in xrange(2)]
        for (u, v, w) in edges:
            adj1[u].append((v, w))
            adj2[v].append((u, w))
        dist1 = dijkstra(adj1, src1)
        dist2 = dijkstra(adj1, src2)
        dist3 = dijkstra(adj2, dest)
        result = min((dist1[i] + dist2[i] + dist3[i] for i in xrange(n)))
        return result if result != float('inf') else -1