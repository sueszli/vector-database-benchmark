import collections

class Solution(object):

    def rootCount(self, edges, guesses, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type edges: List[List[int]]\n        :type guesses: List[List[int]]\n        :type k: int\n        :rtype: int\n        '

        def iter_dfs():
            if False:
                print('Hello World!')
            result = 0
            stk = [(0, -1)]
            while stk:
                (u, p) = stk.pop()
                result += int((p, u) in lookup)
                for v in adj[u]:
                    if v == p:
                        continue
                    stk.append((v, u))
            return result

        def iter_dfs2(curr):
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            stk = [(0, -1, curr)]
            while stk:
                (u, p, curr) = stk.pop()
                if (p, u) in lookup:
                    curr -= 1
                if (u, p) in lookup:
                    curr += 1
                result += int(curr >= k)
                for v in adj[u]:
                    if v == p:
                        continue
                    stk.append((v, u, curr))
            return result
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = {(u, v) for (u, v) in guesses}
        curr = iter_dfs()
        return iter_dfs2(curr)
import collections

class Solution2(object):

    def rootCount(self, edges, guesses, k):
        if False:
            return 10
        '\n        :type edges: List[List[int]]\n        :type guesses: List[List[int]]\n        :type k: int\n        :rtype: int\n        '

        def dfs(u, p):
            if False:
                return 10
            cnt = int((p, u) in lookup)
            for v in adj[u]:
                if v == p:
                    continue
                cnt += dfs(v, u)
            return cnt

        def dfs2(u, p, curr):
            if False:
                print('Hello World!')
            if (p, u) in lookup:
                curr -= 1
            if (u, p) in lookup:
                curr += 1
            cnt = int(curr >= k)
            for v in adj[u]:
                if v == p:
                    continue
                cnt += dfs2(v, u, curr)
            return cnt
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = {(u, v) for (u, v) in guesses}
        curr = dfs(0, -1)
        return dfs2(0, -1, curr)
import collections

class Solution3(object):

    def rootCount(self, edges, guesses, k):
        if False:
            return 10
        '\n        :type edges: List[List[int]]\n        :type guesses: List[List[int]]\n        :type k: int\n        :rtype: int\n        '
        cnt = [0]

        def memoization(u, p):
            if False:
                while True:
                    i = 10
            if (u, p) not in memo:
                memo[u, p] = int((p, u) in lookup)
                for v in adj[u]:
                    if v == p:
                        continue
                    cnt[0] += 1
                    memo[u, p] += memoization(v, u)
            return memo[u, p]
        adj = collections.defaultdict(list)
        for (u, v) in edges:
            adj[u].append(v)
            adj[v].append(u)
        lookup = {(u, v) for (u, v) in guesses}
        memo = {}
        return sum((memoization(i, -1) >= k for i in adj.iterkeys()))