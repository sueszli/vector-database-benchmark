class UnionFind(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.set = range(n)
        self.rank = [0] * n
        self.right = range(n)

    def find_set(self, x):
        if False:
            for i in range(10):
                print('nop')
        stk = []
        while self.set[x] != x:
            stk.append(x)
            x = self.set[x]
        while stk:
            self.set[stk.pop()] = x
        return x

    def union_set(self, x, y):
        if False:
            return 10
        (x, y) = (self.find_set(x), self.find_set(y))
        if x == y:
            return False
        if self.rank[x] > self.rank[y]:
            (x, y) = (y, x)
        self.set[x] = self.set[y]
        if self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        self.right[y] = max(self.right[x], self.right[y])
        return True

    def right_set(self, x):
        if False:
            print('Hello World!')
        return self.right[self.find_set(x)]

class Solution(object):

    def minReverseOperations(self, n, p, banned, k):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type p: int\n        :type banned: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        lookup = [False] * n
        for i in banned:
            lookup[i] = True
        d = 0
        result = [-1] * n
        result[p] = d
        uf = UnionFind(n + 2)
        uf.union_set(p, p + 2)
        q = [p]
        d += 1
        while q:
            new_q = []
            for p in q:
                (left, right) = (2 * max(p - (k - 1), 0) + (k - 1) - p, 2 * min(p + (k - 1), n - 1) - (k - 1) - p)
                p = uf.right_set(left)
                while p <= right:
                    if not lookup[p]:
                        result[p] = d
                        new_q.append(p)
                    uf.union_set(p, p + 2)
                    p = uf.right_set(p)
            q = new_q
            d += 1
        return result
from sortedcontainers import SortedList

class Solution2(object):

    def minReverseOperations(self, n, p, banned, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type p: int\n        :type banned: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        lookup = [False] * n
        for i in banned:
            lookup[i] = True
        d = 0
        result = [-1] * n
        result[p] = d
        sl = [SortedList((i for i in xrange(0, n, 2))), SortedList((i for i in xrange(1, n, 2)))]
        sl[p % 2].remove(p)
        q = [p]
        d += 1
        while q:
            new_q = []
            for p in q:
                (left, right) = (2 * max(p - (k - 1), 0) + (k - 1) - p, 2 * min(p + (k - 1), n - 1) - (k - 1) - p)
                for p in list(sl[left % 2].irange(left, right)):
                    if not lookup[p]:
                        result[p] = d
                        new_q.append(p)
                    sl[left % 2].remove(p)
            q = new_q
            d += 1
        return result