import collections

class Solution(object):

    def numberOfNodes(self, n, queries):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type queries: List[int]\n        :rtype: int\n        '

        def bfs():
            if False:
                while True:
                    i = 10
            result = 0
            q = [(1, 0)]
            while q:
                new_q = []
                for (u, curr) in q:
                    curr ^= cnt[u] % 2
                    result += curr
                    for v in xrange(2 * u, min(2 * u + 1, n) + 1):
                        q.append((v, curr))
                q = new_q
            return result
        cnt = collections.Counter(queries)
        return bfs()
import collections

class Solution2(object):

    def numberOfNodes(self, n, queries):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type queries: List[int]\n        :rtype: int\n        '

        def iter_dfs():
            if False:
                print('Hello World!')
            result = 0
            stk = [(1, 0)]
            while stk:
                (u, curr) = stk.pop()
                curr ^= cnt[u] % 2
                result += curr
                for v in reversed(xrange(2 * u, min(2 * u + 1, n) + 1)):
                    stk.append((v, curr))
            return result
        cnt = collections.Counter(queries)
        return iter_dfs()
import collections

class Solution3(object):

    def numberOfNodes(self, n, queries):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type queries: List[int]\n        :rtype: int\n        '

        def dfs(u, curr):
            if False:
                i = 10
                return i + 15
            curr ^= cnt[u] % 2
            return curr + sum((dfs(v, curr) for v in xrange(2 * u, min(2 * u + 1, n) + 1)))
        cnt = collections.Counter(queries)
        return dfs(1, 0)