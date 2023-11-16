import collections
from functools import partial

class UnionFind(object):

    def __init__(self, n):
        if False:
            return 10
        self.set = range(n)
        self.rank = [0] * n
        self.ancestor = range(n)

    def find_set(self, x):
        if False:
            while True:
                i = 10
        stk = []
        while self.set[x] != x:
            stk.append(x)
            x = self.set[x]
        while stk:
            self.set[stk.pop()] = x
        return x

    def union_set(self, x, y):
        if False:
            while True:
                i = 10
        (x, y) = (self.find_set(x), self.find_set(y))
        if x == y:
            return False
        if self.rank[x] > self.rank[y]:
            (x, y) = (y, x)
        self.set[x] = self.set[y]
        if self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        return True

    def find_ancestor_of_set(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.ancestor[self.find_set(x)]

    def update_ancestor_of_set(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.ancestor[self.find_set(x)] = x

class TreeInfos(object):

    def __init__(self, adj, pairs):
        if False:
            print('Hello World!')

        def preprocess(u, p, w):
            if False:
                return 10
            D[u] = 1 if p == -1 else D[p] + 1
            if w != -1:
                cnt[w] += 1
            CNT[u] = cnt[:]

        def divide(u, p, w):
            if False:
                i = 10
                return i + 15
            stk.append(partial(postprocess, u, w))
            for i in reversed(xrange(len(adj[u]))):
                (v, nw) = adj[u][i]
                if v == p:
                    continue
                stk.append(partial(conquer, v, u))
                stk.append(partial(divide, v, u, nw))
            stk.append(partial(preprocess, u, p, w))

        def conquer(u, p):
            if False:
                while True:
                    i = 10
            uf.union_set(u, p)
            uf.update_ancestor_of_set(p)

        def postprocess(u, w):
            if False:
                while True:
                    i = 10
            lookup[u] = True
            for v in pairs[u]:
                if not lookup[v]:
                    continue
                lca[min(u, v), max(u, v)] = uf.find_ancestor_of_set(v)
            if w != -1:
                cnt[w] -= 1
        N = len(adj)
        (D, uf, lca) = ([0] * N, UnionFind(N), {})
        CNT = [[0] * MAX_W for _ in xrange(N)]
        cnt = [0] * MAX_W
        (stk, lookup) = ([], [False] * N)
        stk.append(partial(divide, 0, -1, -1))
        while stk:
            stk.pop()()
        (self.D, self.lca) = (D, lca)
        self.CNT = CNT
MAX_W = 26

class Solution(object):

    def minOperationsQueries(self, n, edges, queries):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        adj = [[] for _ in xrange(n)]
        for (u, v, w) in edges:
            w -= 1
            adj[u].append((v, w))
            adj[v].append((u, w))
        pairs = collections.defaultdict(set)
        for (a, b) in queries:
            (pairs[a].add(b), pairs[b].add(a))
        tree_infos = TreeInfos(adj, pairs)
        result = [0] * len(queries)
        for (i, (a, b)) in enumerate(queries):
            lca = tree_infos.lca[min(a, b), max(a, b)]
            result[i] = tree_infos.D[a] + tree_infos.D[b] - 2 * tree_infos.D[lca] - max((tree_infos.CNT[a][w] + tree_infos.CNT[b][w] - 2 * tree_infos.CNT[lca][w] for w in xrange(MAX_W)))
        return result
import collections
from functools import partial

class TreeInfos2(object):

    def __init__(self, adj):
        if False:
            for i in range(10):
                print('nop')

        def preprocess(u, p, w):
            if False:
                i = 10
                return i + 15
            D[u] = 1 if p == -1 else D[p] + 1
            if p != -1:
                P[u].append(p)
            i = 0
            while i < len(P[u]) and i < len(P[P[u][i]]):
                P[u].append(P[P[u][i]][i])
                i += 1
            C[0] += 1
            L[u] = C[0]
            if w != -1:
                cnt[w] += 1
            CNT[u] = cnt[:]

        def divide(u, p, w):
            if False:
                print('Hello World!')
            stk.append(partial(postprocess, u, w))
            for i in reversed(xrange(len(adj[u]))):
                (v, nw) = adj[u][i]
                if v == p:
                    continue
                stk.append(partial(divide, v, u, nw))
            stk.append(partial(preprocess, u, p, w))

        def postprocess(u, w):
            if False:
                return 10
            R[u] = C[0]
            if w != -1:
                cnt[w] -= 1
        N = len(adj)
        (L, R, D, P, C) = ([0] * N, [0] * N, [0] * N, [[] for _ in xrange(N)], [-1])
        CNT = [[0] * MAX_W for _ in xrange(N)]
        cnt = [0] * MAX_W
        stk = []
        stk.append(partial(divide, 0, -1, -1))
        while stk:
            stk.pop()()
        assert C[0] == N - 1
        (self.L, self.R, self.D, self.P) = (L, R, D, P)
        self.CNT = CNT

    def is_ancestor(self, a, b):
        if False:
            i = 10
            return i + 15
        return self.L[a] <= self.L[b] <= self.R[b] <= self.R[a]

    def lca(self, a, b):
        if False:
            while True:
                i = 10
        if self.D[a] > self.D[b]:
            (a, b) = (b, a)
        if self.is_ancestor(a, b):
            return a
        for i in reversed(xrange(len(self.P[a]))):
            if i < len(self.P[a]) and (not self.is_ancestor(self.P[a][i], b)):
                a = self.P[a][i]
        return self.P[a][0]
MAX_W = 26

class Solution2(object):

    def minOperationsQueries(self, n, edges, queries):
        if False:
            return 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type queries: List[List[int]]\n        :rtype: List[int]\n        '
        adj = [[] for _ in xrange(n)]
        for (u, v, w) in edges:
            w -= 1
            adj[u].append((v, w))
            adj[v].append((u, w))
        tree_infos = TreeInfos2(adj)
        result = [0] * len(queries)
        for (i, (a, b)) in enumerate(queries):
            lca = tree_infos.lca(a, b)
            result[i] = tree_infos.D[a] + tree_infos.D[b] - 2 * tree_infos.D[lca] - max((tree_infos.CNT[a][w] + tree_infos.CNT[b][w] - 2 * tree_infos.CNT[lca][w] for w in xrange(MAX_W)))
        return result