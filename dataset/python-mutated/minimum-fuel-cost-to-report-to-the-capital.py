class Solution(object):

    def minimumFuelCost(self, roads, seats):
        if False:
            while True:
                i = 10
        '\n        :type roads: List[List[int]]\n        :type seats: int\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                while True:
                    i = 10
            return (a + b - 1) // b

        def iter_dfs():
            if False:
                return 10
            result = 0
            stk = [(1, (0, -1, 0, [1]))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (u, p, d, ret) = args
                    stk.append((3, (d, ret)))
                    for v in adj[u]:
                        if v == p:
                            continue
                        new_ret = [1]
                        stk.append((2, (new_ret, ret)))
                        stk.append((1, (v, u, d + 1, new_ret)))
                elif step == 2:
                    (new_ret, ret) = args
                    ret[0] += new_ret[0]
                elif step == 3:
                    (d, ret) = args
                    if d:
                        result += ceil_divide(ret[0], seats)
            return result
        adj = [[] for _ in xrange(len(roads) + 1)]
        for (u, v) in roads:
            adj[u].append(v)
            adj[v].append(u)
        return iter_dfs()

class Solution(object):

    def minimumFuelCost(self, roads, seats):
        if False:
            print('Hello World!')
        '\n        :type roads: List[List[int]]\n        :type seats: int\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                return 10
            return (a + b - 1) // b

        def dfs(u, p, d):
            if False:
                while True:
                    i = 10
            cnt = 1 + sum((dfs(v, u, d + 1) for v in adj[u] if v != p))
            if d:
                result[0] += ceil_divide(cnt, seats)
            return cnt
        adj = [[] for _ in xrange(len(roads) + 1)]
        for (u, v) in roads:
            adj[u].append(v)
            adj[v].append(u)
        result = [0]
        dfs(0, -1, 0)
        return result[0]