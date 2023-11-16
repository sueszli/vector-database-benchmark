from functools import partial

class TreeInfos(object):

    def __init__(self, children):
        if False:
            while True:
                i = 10

        def preprocess(curr, parent, weight):
            if False:
                for i in range(10):
                    print('nop')
            if parent != -1:
                W[curr].append(weight)
                P[curr].append(parent)
            i = 0
            while i < len(P[curr]) and i < len(P[P[curr][i]]):
                W[curr].append(max(W[curr][i], W[P[curr][i]][i]))
                P[curr].append(P[P[curr][i]][i])
                i += 1
            C[0] += 1
            L[curr] = C[0]

        def divide(curr, parent, weight):
            if False:
                for i in range(10):
                    print('nop')
            stk.append(partial(postprocess, curr))
            for (child, w) in reversed(children[curr]):
                if child == parent:
                    continue
                stk.append(partial(divide, child, curr, w))
            stk.append(partial(preprocess, curr, parent, weight))

        def postprocess(curr):
            if False:
                while True:
                    i = 10
            R[curr] = C[0]
        N = len(children)
        (L, R, P, W, C) = ([0] * N, [0] * N, [[] for _ in xrange(N)], [[] for _ in xrange(N)], [-1])
        for i in xrange(N):
            if L[i]:
                continue
            stk = []
            stk.append(partial(divide, i, -1, 0))
            while stk:
                stk.pop()()
        (self.L, self.R, self.P, self.W) = (L, R, P, W)

    def is_ancestor(self, a, b):
        if False:
            return 10
        return self.L[a] <= self.L[b] <= self.R[b] <= self.R[a]

    def max_weights(self, a, b):
        if False:
            i = 10
            return i + 15

        def binary_lift(a, b):
            if False:
                i = 10
                return i + 15
            w = 0
            for i in reversed(xrange(len(self.P[a]))):
                if i < len(self.P[a]) and (not self.is_ancestor(self.P[a][i], b)):
                    w = max(w, self.W[a][i])
                    a = self.P[a][i]
            return max(w, self.W[a][0])
        w = 0
        if not self.is_ancestor(a, b):
            w = max(w, binary_lift(a, b))
        if not self.is_ancestor(b, a):
            w = max(w, binary_lift(b, a))
        return w

class UnionFind(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
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
            i = 10
            return i + 15
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

class DistanceLimitedPathsExist(object):

    def __init__(self, n, edgeList):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edgeList: List[List[int]]\n        '
        edgeList.sort(key=lambda x: x[2])
        self.__uf = UnionFind(n)
        self.__adj = [[] for _ in xrange(n)]
        for (index, (i, j, weight)) in enumerate(edgeList):
            if not self.__uf.union_set(i, j):
                continue
            self.__adj[i].append((j, weight))
            self.__adj[j].append((i, weight))
        self.__tree_infos = TreeInfos(self.__adj)

    def query(self, p, q, limit):
        if False:
            while True:
                i = 10
        '\n        :type p: int\n        :type q: int\n        :type limit: int\n        :rtype: bool\n        '
        if self.__uf.find_set(p) != self.__uf.find_set(q):
            return False
        return self.__tree_infos.max_weights(p, q) < limit
import collections
import sortedcontainers
import bisect

class SnapshotArray(object):

    def __init__(self, length):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type length: int\n        '
        self.__snaps = collections.defaultdict(lambda : sortedcontainers.SortedList([(0, 0)]))

    def set(self, index, val, snap_id):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type index: int\n        :type val: int\n        :rtype: None\n        '
        i = self.__snaps[index].bisect_left((snap_id, float('-inf')))
        if i != len(self.__snaps[index]) and self.__snaps[index][i][0] == snap_id:
            self.__snaps[index].remove(self.__snaps[index][i])
        self.__snaps[index].add((snap_id, val))

    def get(self, index, snap_id):
        if False:
            return 10
        '\n        :type index: int\n        :type snap_id: int\n        :rtype: int\n        '
        i = self.__snaps[index].bisect_left((snap_id + 1, float('-inf'))) - 1
        return self.__snaps[index][i][1]

class VersionedUnionFind(object):

    def __init__(self, n):
        if False:
            return 10
        self.snap_id = 0
        self.set = SnapshotArray(n)
        for i in xrange(n):
            self.set.set(i, i, self.snap_id)
        self.rank = SnapshotArray(n)

    def find_set(self, x, snap_id):
        if False:
            return 10
        stk = []
        while self.set.get(x, snap_id) != x:
            stk.append(x)
            x = self.set.get(x, snap_id)
        while stk:
            self.set.set(stk.pop(), x, snap_id)
        return x

    def union_set(self, x, y):
        if False:
            return 10
        x_root = self.find_set(x, self.snap_id)
        y_root = self.find_set(y, self.snap_id)
        if x_root == y_root:
            return False
        if self.rank.get(x_root, self.snap_id) < self.rank.get(y_root, self.snap_id):
            self.set.set(x_root, y_root, self.snap_id)
        elif self.rank.get(x_root, self.snap_id) > self.rank.get(y_root, self.snap_id):
            self.set.set(y_root, x_root, self.snap_id)
        else:
            self.set.set(y_root, x_root, self.snap_id)
            self.rank.set(x_root, self.rank.get(x_root, self.snap_id) + 1, self.snap_id)
        return True

    def snap(self):
        if False:
            return 10
        self.snap_id += 1

class DistanceLimitedPathsExist2(object):

    def __init__(self, n, edgeList):
        if False:
            return 10
        '\n        :type n: int\n        :type edgeList: List[List[int]]\n        '
        edgeList.sort(key=lambda x: x[2])
        self.__uf = VersionedUnionFind(n)
        self.__weights = []
        for (index, (i, j, weight)) in enumerate(edgeList):
            if not self.__uf.union_set(i, j):
                continue
            self.__uf.snap()
            self.__weights.append(weight)

    def query(self, p, q, limit):
        if False:
            i = 10
            return i + 15
        '\n        :type p: int\n        :type q: int\n        :type limit: int\n        :rtype: bool\n        '
        snap_id = bisect.bisect_left(self.__weights, limit) - 1
        if snap_id == -1:
            return False
        return self.__uf.find_set(p, snap_id) == self.__uf.find_set(q, snap_id)