class Solution(object):

    def buildMatrix(self, k, rowConditions, colConditions):
        if False:
            print('Hello World!')
        '\n        :type k: int\n        :type rowConditions: List[List[int]]\n        :type colConditions: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def topological_sort(conditions):
            if False:
                while True:
                    i = 10
            adj = [[] for _ in xrange(k)]
            in_degree = [0] * k
            for (u, v) in conditions:
                u -= 1
                v -= 1
                adj[u].append(v)
                in_degree[v] += 1
            result = []
            q = [u for u in xrange(k) if not in_degree[u]]
            while q:
                new_q = []
                for u in q:
                    result.append(u)
                    for v in adj[u]:
                        in_degree[v] -= 1
                        if in_degree[v]:
                            continue
                        new_q.append(v)
                q = new_q
            return result
        row_order = topological_sort(rowConditions)
        if len(row_order) != k:
            return []
        col_order = topological_sort(colConditions)
        if len(col_order) != k:
            return []
        row_idx = {x: i for (i, x) in enumerate(row_order)}
        col_idx = {x: i for (i, x) in enumerate(col_order)}
        result = [[0] * k for _ in xrange(k)]
        for i in xrange(k):
            result[row_idx[i]][col_idx[i]] = i + 1
        return result