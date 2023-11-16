class Solution(object):

    def minimumTotalPrice(self, n, edges, price, trips):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type price: List[int]\n        :type trips: List[List[int]]\n        :rtype: int\n        '

        def iter_dfs(u, target):
            if False:
                while True:
                    i = 10
            stk = [(1, (u, -1))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (u, p) = args
                    lookup[u] += 1
                    if u == target:
                        return
                    stk.append((2, (u,)))
                    for v in reversed(adj[u]):
                        if v == p:
                            continue
                        stk.append((1, (v, u)))
                elif step == 2:
                    u = args[0]
                    lookup[u] -= 1
            lookup[u] += 1
            if u == target:
                return True
            for v in adj[u]:
                if v == p:
                    continue
                if dfs(v, u, target):
                    return True
            lookup[u] -= 1
            return False

        def iter_dfs2():
            if False:
                for i in range(10):
                    print('nop')
            result = [price[0] * lookup[0], price[0] // 2 * lookup[0]]
            stk = [(1, (0, -1, result))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (u, p, ret) = args
                    for v in reversed(adj[u]):
                        if v == p:
                            continue
                        new_ret = [price[v] * lookup[v], price[v] // 2 * lookup[v]]
                        stk.append((2, (new_ret, ret)))
                        stk.append((1, (v, u, new_ret)))
                elif step == 2:
                    (new_ret, ret) = args
                    ret[0] += min(new_ret)
                    ret[1] += new_ret[0]
            return min(result)
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = [0] * n
        for (u, v) in trips:
            iter_dfs(u, v)
        return iter_dfs2()

class Solution2(object):

    def minimumTotalPrice(self, n, edges, price, trips):
        if False:
            return 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type price: List[int]\n        :type trips: List[List[int]]\n        :rtype: int\n        '

        def dfs(u, p, target):
            if False:
                print('Hello World!')
            lookup[u] += 1
            if u == target:
                return True
            for v in adj[u]:
                if v == p:
                    continue
                if dfs(v, u, target):
                    return True
            lookup[u] -= 1
            return False

        def dfs2(u, p):
            if False:
                return 10
            (full, half) = (price[u] * lookup[u], price[u] // 2 * lookup[u])
            for v in adj[u]:
                if v == p:
                    continue
                (f, h) = dfs2(v, u)
                full += min(f, h)
                half += f
            return (full, half)
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = [0] * n
        for (u, v) in trips:
            dfs(u, -1, v)
        return min(dfs2(0, -1))