class Solution(object):

    def getAncestors(self, n, edges):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def iter_dfs(adj, i, result):
            if False:
                print('Hello World!')
            lookup = [False] * len(adj)
            stk = [i]
            while stk:
                u = stk.pop()
                for v in reversed(adj[u]):
                    if lookup[v]:
                        continue
                    lookup[v] = True
                    stk.append(v)
                    result[v].append(i)
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
        result = [[] for _ in xrange(n)]
        for u in xrange(n):
            iter_dfs(adj, u, result)
        return result

class Solution2(object):

    def getAncestors(self, n, edges):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def bfs(adj, i, result):
            if False:
                for i in range(10):
                    print('nop')
            lookup = [False] * len(adj)
            q = [i]
            lookup[i] = True
            while q:
                new_q = []
                for u in q:
                    for v in adj[u]:
                        if lookup[v]:
                            continue
                        lookup[v] = True
                        new_q.append(v)
                        result[i].append(v)
                q = new_q
            result[i].sort()
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[v].append(u)
        result = [[] for _ in xrange(n)]
        for u in xrange(n):
            bfs(adj, u, result)
        return result

class Solution3(object):

    def getAncestors(self, n, edges):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[List[int]]\n        '
        result = [set() for _ in xrange(n)]
        in_degree = [0] * n
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            in_degree[v] += 1
            result[v].add(u)
        q = [u for (u, d) in enumerate(in_degree) if not d]
        while q:
            new_q = []
            for u in q:
                for v in adj[u]:
                    result[v].update(result[u])
                    in_degree[v] -= 1
                    if not in_degree[v]:
                        new_q.append(v)
            q = new_q
        return [sorted(s) for s in result]