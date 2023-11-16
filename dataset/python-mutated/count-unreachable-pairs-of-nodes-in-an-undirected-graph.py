class Solution(object):

    def countPairs(self, n, edges):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def bfs(adj, u, lookup):
            if False:
                return 10
            q = [u]
            lookup[u] = 1
            result = 1
            while q:
                new_q = []
                for u in q:
                    for v in adj[u]:
                        if lookup[v]:
                            continue
                        lookup[v] = 1
                        result += 1
                        new_q.append(v)
                q = new_q
            return result
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = [0] * n
        result = 0
        for u in xrange(n):
            if lookup[u]:
                continue
            cnt = bfs(adj, u, lookup)
            result += cnt * (n - cnt)
            n -= cnt
        return result