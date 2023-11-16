class Solution(object):

    def maximumScoreAfterOperations(self, edges, values):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type edges: List[List[int]]\n        :type values: List[int]\n        :rtype: int\n        '

        def iter_dfs():
            if False:
                i = 10
                return i + 15
            dp = [0] * len(values)
            stk = [(1, 0, -1)]
            while stk:
                (step, u, p) = stk.pop()
                if step == 1:
                    if len(adj[u]) == (1 if u else 0):
                        dp[u] = values[u]
                        continue
                    stk.append((2, u, p))
                    for v in reversed(adj[u]):
                        if v != p:
                            stk.append((1, v, u))
                elif step == 2:
                    dp[u] = min(sum((dp[v] for v in adj[u] if v != p)), values[u])
            return dp[0]
        adj = [[] for _ in xrange(len(values))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return sum(values) - iter_dfs()

class Solution2(object):

    def maximumScoreAfterOperations(self, edges, values):
        if False:
            while True:
                i = 10
        '\n        :type edges: List[List[int]]\n        :type values: List[int]\n        :rtype: int\n        '

        def dfs(u, p):
            if False:
                print('Hello World!')
            if len(adj[u]) == (1 if u else 0):
                return values[u]
            return min(sum((dfs(v, u) for v in adj[u] if v != p)), values[u])
        adj = [[] for _ in xrange(len(values))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return sum(values) - dfs(0, -1)