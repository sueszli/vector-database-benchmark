class Solution(object):

    def maxOutput(self, n, edges, price):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type price: List[int]\n        :rtype: int\n        '

        def iter_dfs():
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            stk = [(1, (0, -1, [price[0], 0]))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (u, p, ret) = args
                    stk.append((2, (u, p, ret, 0)))
                elif step == 2:
                    (u, p, ret, i) = args
                    if i == len(adj[u]):
                        continue
                    stk.append((2, (u, p, ret, i + 1)))
                    v = adj[u][i]
                    if v == p:
                        continue
                    new_ret = [price[v], 0]
                    stk.append((3, (u, new_ret, ret)))
                    stk.append((1, (v, u, new_ret)))
                elif step == 3:
                    (u, new_ret, ret) = args
                    result = max(result, ret[0] + new_ret[1], ret[1] + new_ret[0])
                    ret[0] = max(ret[0], new_ret[0] + price[u])
                    ret[1] = max(ret[1], new_ret[1] + price[u])
            return result
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        return iter_dfs()

class Solution2(object):

    def maxOutput(self, n, edges, price):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type price: List[int]\n        :rtype: int\n        '

        def dfs(u, p):
            if False:
                i = 10
                return i + 15
            dp = [price[u], 0]
            for v in adj[u]:
                if v == p:
                    continue
                new_dp = dfs(v, u)
                result[0] = max(result[0], dp[0] + new_dp[1], dp[1] + new_dp[0])
                dp[0] = max(dp[0], new_dp[0] + price[u])
                dp[1] = max(dp[1], new_dp[1] + price[u])
            return dp
        result = [0]
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        dfs(0, -1)
        return result[0]

class Solution3(object):

    def maxOutput(self, n, edges, price):
        if False:
            return 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type price: List[int]\n        :rtype: int\n        '

        def iter_dfs():
            if False:
                for i in range(10):
                    print('nop')
            dp = [0] * n
            stk = [(1, 0, -1)]
            while stk:
                (step, u, p) = stk.pop()
                if step == 1:
                    stk.append((2, u, p))
                    for v in adj[u]:
                        if v == p:
                            continue
                        stk.append((1, v, u))
                elif step == 2:
                    dp[u] = price[u]
                    for v in adj[u]:
                        if v == p:
                            continue
                        dp[u] = max(dp[u], dp[v] + price[u])
            return dp

        def iter_dfs2():
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            stk = [(0, -1, 0)]
            while stk:
                (u, p, curr) = stk.pop()
                result = max(result, curr, dp[u] - price[u])
                top2 = [[curr, p], [0, -1]]
                for v in adj[u]:
                    if v == p:
                        continue
                    curr = [dp[v], v]
                    for i in xrange(len(top2)):
                        if curr > top2[i]:
                            (top2[i], curr) = (curr, top2[i])
                for v in adj[u]:
                    if v == p:
                        continue
                    stk.append((v, u, (top2[0][0] if top2[0][1] != v else top2[1][0]) + price[u]))
            return result
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        dp = iter_dfs()
        return iter_dfs2()

class Solution4(object):

    def maxOutput(self, n, edges, price):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type price: List[int]\n        :rtype: int\n        '

        def dfs(u, p):
            if False:
                while True:
                    i = 10
            dp[u] = price[u]
            for v in adj[u]:
                if v == p:
                    continue
                dp[u] = max(dp[u], dfs(v, u) + price[u])
            return dp[u]

        def dfs2(u, p, curr):
            if False:
                print('Hello World!')
            result[0] = max(result[0], curr, dp[u] - price[u])
            top2 = [[curr, p], [0, -1]]
            for v in adj[u]:
                if v == p:
                    continue
                curr = [dp[v], v]
                for i in xrange(len(top2)):
                    if curr > top2[i]:
                        (top2[i], curr) = (curr, top2[i])
            for v in adj[u]:
                if v == p:
                    continue
                dfs2(v, u, (top2[0][0] if top2[0][1] != v else top2[1][0]) + price[u])
        result = [0]
        dp = [0] * n
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        dfs(0, -1)
        dfs2(0, -1, 0)
        return result[0]