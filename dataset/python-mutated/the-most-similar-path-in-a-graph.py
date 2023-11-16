class Solution(object):

    def mostSimilar(self, n, roads, names, targetPath):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type roads: List[List[int]]\n        :type names: List[str]\n        :type targetPath: List[str]\n        :rtype: List[int]\n        '
        adj = [[] for _ in xrange(n)]
        for (u, v) in roads:
            adj[u].append(v)
            adj[v].append(u)
        dp = [[0] * n for _ in xrange(len(targetPath) + 1)]
        for i in xrange(1, len(targetPath) + 1):
            for v in xrange(n):
                dp[i][v] = (names[v] != targetPath[i - 1]) + min((dp[i - 1][u] for u in adj[v]))
        path = [dp[-1].index(min(dp[-1]))]
        for i in reversed(xrange(2, len(targetPath) + 1)):
            for u in adj[path[-1]]:
                if dp[i - 1][u] + (names[path[-1]] != targetPath[i - 1]) == dp[i][path[-1]]:
                    path.append(u)
                    break
        return path[::-1]