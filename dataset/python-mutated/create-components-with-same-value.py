class Solution(object):

    def componentValue(self, nums, edges):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def bfs(target):
            if False:
                while True:
                    i = 10
            total = nums[:]
            lookup = [len(adj[u]) for u in xrange(len(adj))]
            q = [u for u in xrange(len(adj)) if lookup[u] == 1]
            while q:
                new_q = []
                for u in q:
                    if total[u] > target:
                        return False
                    if total[u] == target:
                        total[u] = 0
                    for v in adj[u]:
                        total[v] += total[u]
                        lookup[v] -= 1
                        if lookup[v] == 1:
                            new_q.append(v)
                q = new_q
            return True
        result = 0
        adj = [[] for _ in xrange(len(nums))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        total = sum(nums)
        for cnt in reversed(xrange(2, len(nums) + 1)):
            if total % cnt == 0 and bfs(total // cnt):
                return cnt - 1
        return 0

class Solution2(object):

    def componentValue(self, nums, edges):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def iter_dfs(target):
            if False:
                i = 10
                return i + 15
            total = nums[:]
            stk = [(1, (0, -1))]
            while stk:
                (step, (u, p)) = stk.pop()
                if step == 1:
                    stk.append((2, (u, p)))
                    for v in adj[u]:
                        if v == p:
                            continue
                        stk.append((1, (v, u)))
                elif step == 2:
                    for v in adj[u]:
                        if v == p:
                            continue
                        total[u] += total[v]
                    if total[u] == target:
                        total[u] = 0
            return total[0]
        result = 0
        adj = [[] for _ in xrange(len(nums))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        total = sum(nums)
        for cnt in reversed(xrange(2, len(nums) + 1)):
            if total % cnt == 0 and iter_dfs(total // cnt) == 0:
                return cnt - 1
        return 0

class Solution3(object):

    def componentValue(self, nums, edges):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def dfs(u, p, target):
            if False:
                for i in range(10):
                    print('nop')
            total = nums[u]
            for v in adj[u]:
                if v == p:
                    continue
                total += dfs(v, u, target)
            return total if total != target else 0
        result = 0
        adj = [[] for _ in xrange(len(nums))]
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        total = sum(nums)
        for cnt in reversed(xrange(2, len(nums) + 1)):
            if total % cnt == 0 and dfs(0, -1, total // cnt) == 0:
                return cnt - 1
        return 0