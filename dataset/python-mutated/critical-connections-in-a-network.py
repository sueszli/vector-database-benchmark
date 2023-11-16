class Solution(object):

    def criticalConnections(self, n, connections):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type connections: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def dfs(edges, parent, u, idx, lowlinks, lookup, result):
            if False:
                return 10
            if lookup[u]:
                return
            lookup[u] = True
            curr_idx = lowlinks[u] = idx[0]
            idx[0] += 1
            for v in edges[u]:
                if v == parent:
                    continue
                dfs(edges, u, v, idx, lowlinks, lookup, result)
                lowlinks[u] = min(lowlinks[u], lowlinks[v])
                if lowlinks[v] > curr_idx:
                    result.append([u, v])
        edges = [[] for _ in xrange(n)]
        (idx, lowlinks, lookup) = ([0], [0] * n, [False] * n)
        result = []
        for (u, v) in connections:
            edges[u].append(v)
            edges[v].append(u)
        dfs(edges, -1, 0, idx, lowlinks, lookup, result)
        return result