class Solution(object):

    def minEdgeReversals(self, n, edges):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '

        def iter_dfs1():
            if False:
                return 10
            result = 0
            stk = [(0, -1)]
            while stk:
                (u, p) = stk.pop()
                for v in adj[u].iterkeys():
                    if v == p:
                        continue
                    result += adj[u][v]
                    stk.append((v, u))
            return result

        def iter_dfs2(curr):
            if False:
                print('Hello World!')
            result = [-1] * n
            stk = [(0, curr)]
            while stk:
                (u, curr) = stk.pop()
                result[u] = curr
                for v in adj[u].iterkeys():
                    if result[v] == -1:
                        stk.append((v, curr - adj[u][v] + adj[v][u]))
            return result
        adj = collections.defaultdict(dict)
        for (u, v) in edges:
            adj[u][v] = 0
            adj[v][u] = 1
        return iter_dfs2(iter_dfs1())

class Solution2(object):

    def minEdgeReversals(self, n, edges):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: List[int]\n        '

        def dfs1(u, p):
            if False:
                while True:
                    i = 10
            return sum((adj[u][v] + dfs1(v, u) for v in adj[u] if v != p))

        def dfs2(u, curr):
            if False:
                print('Hello World!')
            result[u] = curr
            for v in adj[u]:
                if result[v] == -1:
                    dfs2(v, curr - adj[u][v] + adj[v][u])
        adj = collections.defaultdict(dict)
        for (u, v) in edges:
            adj[u][v] = 0
            adj[v][u] = 1
        result = [-1] * n
        dfs2(0, dfs1(0, -1))
        return result