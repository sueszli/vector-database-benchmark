class Solution(object):

    def findCircleNum(self, M):
        if False:
            return 10
        '\n        :type M: List[List[int]]\n        :rtype: int\n        '

        class UnionFind(object):

            def __init__(self, n):
                if False:
                    i = 10
                    return i + 15
                self.set = range(n)
                self.count = n

            def find_set(self, x):
                if False:
                    print('Hello World!')
                if self.set[x] != x:
                    self.set[x] = self.find_set(self.set[x])
                return self.set[x]

            def union_set(self, x, y):
                if False:
                    while True:
                        i = 10
                (x_root, y_root) = map(self.find_set, (x, y))
                if x_root != y_root:
                    self.set[min(x_root, y_root)] = max(x_root, y_root)
                    self.count -= 1
        circles = UnionFind(len(M))
        for i in xrange(len(M)):
            for j in xrange(len(M)):
                if M[i][j] and i != j:
                    circles.union_set(i, j)
        return circles.count