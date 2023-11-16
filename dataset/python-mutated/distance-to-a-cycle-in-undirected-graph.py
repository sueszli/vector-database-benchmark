class Solution(object):

    def distanceToCycle(self, n, edges):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '

        def cycle(parent, v, u):
            if False:
                while True:
                    i = 10
            result = [parent[v], v]
            while u != parent[v]:
                result.append(u)
                u = parent[u]
            return result

        def iter_dfs(adj):
            if False:
                while True:
                    i = 10
            stk = [0]
            parent = [-2] * len(adj)
            parent[0] = -1
            while stk:
                u = stk.pop()
                for v in reversed(adj[u]):
                    if parent[v] != -2:
                        if v == parent[u]:
                            continue
                        return cycle(parent, v, u)
                    parent[v] = u
                    stk.append(v)

        def bfs(adj, q):
            if False:
                for i in range(10):
                    print('nop')
            result = [-1] * n
            for x in q:
                result[x] = 0
            d = 1
            while q:
                new_q = []
                for u in q:
                    for v in adj[u]:
                        if result[v] != -1:
                            continue
                        result[v] = d
                        new_q.append(v)
                q = new_q
                d += 1
            return result
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return bfs(adj, iter_dfs(adj))