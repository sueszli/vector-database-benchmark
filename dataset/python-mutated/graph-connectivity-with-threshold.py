class UnionFind(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.set = range(n)
        self.rank = [0] * n

    def find_set(self, x):
        if False:
            print('Hello World!')
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
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        if self.rank[x_root] < self.rank[y_root]:
            self.set[x_root] = y_root
        elif self.rank[x_root] > self.rank[y_root]:
            self.set[y_root] = x_root
        else:
            self.set[y_root] = x_root
            self.rank[x_root] += 1
        return True

class Solution(object):

    def areConnected(self, n, threshold, queries):
        if False:
            return 10
        '\n        :type n: int\n        :type threshold: int\n        :type queries: List[List[int]]\n        :rtype: List[bool]\n        '
        union_find = UnionFind(n)
        for i in xrange(threshold + 1, n + 1):
            for j in xrange(2 * i, n + 1, i):
                union_find.union_set(i - 1, j - 1)
        return [union_find.find_set(q[0] - 1) == union_find.find_set(q[1] - 1) for q in queries]