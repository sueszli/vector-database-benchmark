class UnionFind(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.set = range(n)
        self.count = n

    def find_set(self, x):
        if False:
            return 10
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            return 10
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[max(x_root, y_root)] = min(x_root, y_root)
        self.count -= 1
        return True

class Solution(object):

    def earliestAcq(self, logs, N):
        if False:
            print('Hello World!')
        '\n        :type logs: List[List[int]]\n        :type N: int\n        :rtype: int\n        '
        logs.sort()
        union_find = UnionFind(N)
        for (t, a, b) in logs:
            union_find.union_set(a, b)
            if union_find.count == 1:
                return t
        return -1