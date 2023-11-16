class Solution(object):

    def countCompleteComponents(self, n, edges):
        if False:
            return 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def bfs(u):
            if False:
                return 10
            if lookup[u]:
                return False
            v_cnt = e_cnt = 0
            lookup[u] = True
            q = [u]
            while q:
                new_q = []
                v_cnt += len(q)
                for u in q:
                    e_cnt += len(adj[u])
                    for v in adj[u]:
                        if lookup[v]:
                            continue
                        lookup[v] = True
                        new_q.append(v)
                q = new_q
            return v_cnt * (v_cnt - 1) == e_cnt
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = [False] * n
        return sum((bfs(u) for u in xrange(n) if not lookup[u]))