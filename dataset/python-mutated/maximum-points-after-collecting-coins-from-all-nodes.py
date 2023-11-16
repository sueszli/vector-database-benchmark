class Solution(object):

    def maximumPoints(self, edges, coins, k):
        if False:
            return 10
        '\n        :type edges: List[List[int]]\n        :type coins: List[int]\n        :type k: int\n        :rtype: int\n        '
        NEG_INF = float('-inf')

        def dfs(u, p, base):
            if False:
                while True:
                    i = 10
            if base >= max_base:
                return 0
            if lookup[u] & base:
                return NEG_INF
            lookup[u] |= base
            return max(coins[u] // base - k + sum((dfs(v, u, base) for v in adj[u] if v != p)), coins[u] // (base << 1) + sum((dfs(v, u, base << 1) for v in adj[u] if v != p)) if coins[u] // base - k < coins[u] // (base * 2) else NEG_INF)
        adj = [[] for _ in xrange(len(coins))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        max_base = 1 << max(coins).bit_length()
        lookup = [0] * len(coins)
        return dfs(0, -1, 1)

class Solution2(object):

    def maximumPoints(self, edges, coins, k):
        if False:
            return 10
        '\n        :type edges: List[List[int]]\n        :type coins: List[int]\n        :type k: int\n        :rtype: int\n        '

        def memoization(u, p, d):
            if False:
                return 10
            if d >= max_d:
                return 0
            if lookup[u][d] is None:
                lookup[u][d] = max((coins[u] >> d) - k + sum((memoization(v, u, d) for v in adj[u] if v != p)), (coins[u] >> d + 1) + sum((memoization(v, u, d + 1) for v in adj[u] if v != p)))
            return lookup[u][d]
        adj = [[] for _ in xrange(len(coins))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        max_d = max(coins).bit_length()
        lookup = [[None] * max_d for _ in xrange(len(coins))]
        return memoization(0, -1, 0)