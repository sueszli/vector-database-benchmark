class Solution(object):

    def maxKDivisibleComponents(self, n, edges, values, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type values: List[int]\n        :type k: int\n        :rtype: int\n        '

        def bfs():
            if False:
                print('Hello World!')
            result = 0
            dp = [x % k for x in values]
            cnt = [len(adj[u]) for u in xrange(len(adj))]
            q = [u for u in xrange(n) if cnt[u] == 1]
            while q:
                new_q = []
                for u in q:
                    if not dp[u]:
                        result += 1
                    for v in adj[u]:
                        dp[v] = (dp[v] + dp[u]) % k
                        cnt[v] -= 1
                        if cnt[v] == 1:
                            new_q.append(v)
                q = new_q
            return max(result, 1)
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return bfs()