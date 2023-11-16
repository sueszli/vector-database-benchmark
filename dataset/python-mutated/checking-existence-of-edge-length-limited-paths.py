class UnionFind(object):

    def __init__(self, n):
        if False:
            for i in range(10):
                print('nop')
        self.set = range(n)
        self.rank = [0] * n

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
            for i in range(10):
                print('nop')
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

    def distanceLimitedPathsExist(self, n, edgeList, queries):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edgeList: List[List[int]]\n        :type queries: List[List[int]]\n        :rtype: List[bool]\n        '
        for (i, q) in enumerate(queries):
            q.append(i)
        edgeList.sort(key=lambda x: x[2])
        queries.sort(key=lambda x: x[2])
        union_find = UnionFind(n)
        result = [False] * len(queries)
        curr = 0
        for (u, v, w, i) in queries:
            while curr < len(edgeList) and edgeList[curr][2] < w:
                union_find.union_set(edgeList[curr][0], edgeList[curr][1])
                curr += 1
            result[i] = union_find.find_set(u) == union_find.find_set(v)
        return result