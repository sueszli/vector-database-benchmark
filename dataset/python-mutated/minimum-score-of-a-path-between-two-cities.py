class Solution(object):

    def minScore(self, n, roads):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type roads: List[List[int]]\n        :rtype: int\n        '

        def bfs():
            if False:
                print('Hello World!')
            lookup = [False] * len(adj)
            q = [0]
            lookup[0] = True
            while q:
                new_q = []
                for u in q:
                    for (v, _) in adj[u]:
                        if lookup[v]:
                            continue
                        lookup[v] = True
                        new_q.append(v)
                q = new_q
            return lookup
        adj = [[] for _ in xrange(n)]
        for (u, v, w) in roads:
            adj[u - 1].append((v - 1, w))
            adj[v - 1].append((u - 1, w))
        lookup = bfs()
        return min((w for (u, _, w) in roads if lookup[u - 1]))