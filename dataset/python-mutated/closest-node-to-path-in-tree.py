import collections
from functools import partial

class UnionFind(object):

    def __init__(self, n):
        if False:
            print('Hello World!')
        self.set = range(n)
        self.rank = [0] * n
        self.ancestor = range(n)

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
        return True

    def find_ancestor_of_set(self, x):
        if False:
            print('Hello World!')
        return self.ancestor[self.find_set(x)]

    def update_ancestor_of_set(self, x):
        if False:
            i = 10
            return i + 15
        self.ancestor[self.find_set(x)] = x

class TreeInfos(object):

    def __init__(self, children, pairs):
        if False:
            while True:
                i = 10

        def preprocess(curr, parent):
            if False:
                i = 10
                return i + 15
            D[curr] = 1 if parent == -1 else D[parent] + 1

        def divide(curr, parent):
            if False:
                return 10
            stk.append(partial(postprocess, curr))
            for i in reversed(xrange(len(children[curr]))):
                child = children[curr][i]
                if child == parent:
                    continue
                stk.append(partial(conquer, child, curr))
                stk.append(partial(divide, child, curr))
            stk.append(partial(preprocess, curr, parent))

        def conquer(curr, parent):
            if False:
                return 10
            uf.union_set(curr, parent)
            uf.update_ancestor_of_set(parent)

        def postprocess(u):
            if False:
                while True:
                    i = 10
            lookup[u] = True
            for v in pairs[u]:
                if not lookup[v]:
                    continue
                lca[min(u, v), max(u, v)] = uf.find_ancestor_of_set(v)
        N = len(children)
        (D, uf, lca) = ([0] * N, UnionFind(N), {})
        (stk, lookup) = ([], [False] * N)
        stk.append(partial(divide, 0, -1))
        while stk:
            stk.pop()()
        (self.D, self.lca) = (D, lca)

class Solution(object):

    def closestNode(self, n, edges, query):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type query: List[List[int]]\n        :rtype: List[int]\n        '
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            (adj[u].append(v), adj[v].append(u))
        pairs = collections.defaultdict(set)
        for (start, end, node) in query:
            (pairs[start].add(end), pairs[end].add(start))
            (pairs[start].add(node), pairs[node].add(start))
            (pairs[end].add(node), pairs[node].add(end))
        tree_infos = TreeInfos(adj, pairs)
        return [max((tree_infos.lca[min(x, y), max(x, y)] for (x, y) in ((start, end), (start, node), (end, node))), key=lambda x: tree_infos.D[x]) for (start, end, node) in query]
from functools import partial

class TreeInfos2(object):

    def __init__(self, children):
        if False:
            while True:
                i = 10

        def preprocess(curr, parent):
            if False:
                i = 10
                return i + 15
            D[curr] = 1 if parent == -1 else D[parent] + 1
            if parent != -1:
                P[curr].append(parent)
            i = 0
            while i < len(P[curr]) and i < len(P[P[curr][i]]):
                P[curr].append(P[P[curr][i]][i])
                i += 1
            C[0] += 1
            L[curr] = C[0]

        def divide(curr, parent):
            if False:
                while True:
                    i = 10
            stk.append(partial(postprocess, curr))
            for i in reversed(xrange(len(children[curr]))):
                child = children[curr][i]
                if child == parent:
                    continue
                stk.append(partial(divide, child, curr))
            stk.append(partial(preprocess, curr, parent))

        def postprocess(curr):
            if False:
                for i in range(10):
                    print('nop')
            R[curr] = C[0]
        N = len(children)
        (L, R, D, P, C) = ([0] * N, [0] * N, [0] * N, [[] for _ in xrange(N)], [-1])
        stk = []
        stk.append(partial(divide, 0, -1))
        while stk:
            stk.pop()()
        assert C[0] == N - 1
        (self.L, self.R, self.D, self.P) = (L, R, D, P)

    def is_ancestor(self, a, b):
        if False:
            while True:
                i = 10
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

class Solution2(object):

    def closestNode(self, n, edges, query):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type query: List[List[int]]\n        :rtype: List[int]\n        '
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            (adj[u].append(v), adj[v].append(u))
        tree_infos = TreeInfos2(adj)
        return [max((tree_infos.lca(x, y) for (x, y) in ((start, end), (start, node), (end, node))), key=lambda x: tree_infos.D[x]) for (start, end, node) in query]
from functools import partial

class TreeInfos3(object):

    def __init__(self, children):
        if False:
            for i in range(10):
                print('nop')

        def preprocess(curr, parent):
            if False:
                while True:
                    i = 10
            D[curr] = 1 if parent == -1 else D[parent] + 1
            P[curr] = parent

        def divide(curr, parent):
            if False:
                for i in range(10):
                    print('nop')
            for i in reversed(xrange(len(children[curr]))):
                child = children[curr][i]
                if child == parent:
                    continue
                stk.append(partial(divide, child, curr))
            stk.append(partial(preprocess, curr, parent))
        N = len(children)
        (D, P) = ([0] * N, [0] * N)
        stk = []
        stk.append(partial(divide, 0, -1))
        while stk:
            stk.pop()()
        (self.D, self.P) = (D, P)

    def lca(self, a, b):
        if False:
            for i in range(10):
                print('nop')
        while self.D[a] > self.D[b]:
            a = self.P[a]
        while self.D[a] < self.D[b]:
            b = self.P[b]
        while a != b:
            (a, b) = (self.P[a], self.P[b])
        return a

class Solution3(object):

    def closestNode(self, n, edges, query):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type query: List[List[int]]\n        :rtype: List[int]\n        '
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            (adj[u].append(v), adj[v].append(u))
        tree_infos = TreeInfos3(adj)
        return [max((tree_infos.lca(x, y) for (x, y) in ((start, end), (start, node), (end, node))), key=lambda x: tree_infos.D[x]) for (start, end, node) in query]

class Solution4(object):

    def closestNode(self, n, edges, query):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type query: List[List[int]]\n        :rtype: List[int]\n        '

        def bfs(adj, root):
            if False:
                return 10
            dist = [len(adj)] * len(adj)
            q = [root]
            dist[root] = 0
            d = 0
            while q:
                new_q = []
                for u in q:
                    for v in adj[u]:
                        if d + 1 >= dist[v]:
                            continue
                        dist[v] = d + 1
                        new_q.append(v)
                q = new_q
                d += 1
            return dist
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            (adj[u].append(v), adj[v].append(u))
        dist = [bfs(adj, i) for i in xrange(n)]
        result = []
        for (start, end, node) in query:
            x = end
            while start != end:
                if dist[node][start] < dist[node][x]:
                    x = start
                start = next((u for u in adj[start] if dist[u][end] < dist[start][end]))
            result.append(x)
        return result

class Solution5(object):

    def closestNode(self, n, edges, query):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :type query: List[List[int]]\n        :rtype: List[int]\n        '

        def bfs(adj, root):
            if False:
                i = 10
                return i + 15
            dist = [len(adj)] * len(adj)
            q = [root]
            dist[root] = 0
            d = 0
            while q:
                new_q = []
                for u in q:
                    for v in adj[u]:
                        if d + 1 >= dist[v]:
                            continue
                        dist[v] = d + 1
                        new_q.append(v)
                q = new_q
                d += 1
            return dist
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            (adj[u].append(v), adj[v].append(u))
        dist = [bfs(adj, i) for i in xrange(n)]
        return [max((i for i in xrange(n) if dist[start][node] + dist[node][end] - 2 * dist[node][i] == dist[start][i] + dist[i][end]), key=lambda x: dist[node][x]) for (start, end, node) in query]