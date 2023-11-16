class UnionFind(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
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
        if x_root == y_root:
            return False
        self.set[max(x_root, y_root)] = min(x_root, y_root)
        self.count -= 1
        return True

class Solution(object):

    def makeConnected(self, n, connections):
        if False:
            return 10
        '\n        :type n: int\n        :type connections: List[List[int]]\n        :rtype: int\n        '
        if len(connections) < n - 1:
            return -1
        union_find = UnionFind(n)
        for (i, j) in connections:
            union_find.union_set(i, j)
        return union_find.count - 1
import collections

class Solution2(object):

    def makeConnected(self, n, connections):
        if False:
            return 10
        '\n        :type n: int\n        :type connections: List[List[int]]\n        :rtype: int\n        '

        def dfs(i, lookup):
            if False:
                print('Hello World!')
            if i in lookup:
                return 0
            lookup.add(i)
            if i in G:
                for j in G[i]:
                    dfs(j, lookup)
            return 1
        if len(connections) < n - 1:
            return -1
        G = collections.defaultdict(list)
        for (i, j) in connections:
            G[i].append(j)
            G[j].append(i)
        lookup = set()
        return sum((dfs(i, lookup) for i in xrange(n))) - 1