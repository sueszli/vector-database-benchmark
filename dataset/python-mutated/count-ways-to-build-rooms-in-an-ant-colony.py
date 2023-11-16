class Solution(object):

    def waysToBuildRooms(self, prevRoom):
        if False:
            while True:
                i = 10
        '\n        :type prevRoom: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        fact = [1, 1]
        inv = [0, 1]
        inv_fact = [1, 1]

        def nCr(n, k):
            if False:
                for i in range(10):
                    print('nop')
            while len(inv) <= n:
                fact.append(fact[-1] * len(inv) % MOD)
                inv.append(inv[MOD % len(inv)] * (MOD - MOD // len(inv)) % MOD)
                inv_fact.append(inv_fact[-1] * inv[-1] % MOD)
            return fact[n] * inv_fact[n - k] % MOD * inv_fact[k] % MOD

        def dfs(adj, curr):
            if False:
                for i in range(10):
                    print('nop')
            (total_ways, total_cnt) = (1, 0)
            for child in adj[curr]:
                (ways, cnt) = dfs(adj, child)
                total_cnt += cnt
                total_ways = total_ways * ways % MOD * nCr(total_cnt, cnt) % MOD
            return (total_ways, total_cnt + 1)
        adj = [[] for _ in xrange(len(prevRoom))]
        for i in xrange(1, len(prevRoom)):
            adj[prevRoom[i]].append(i)
        return dfs(adj, 0)[0]