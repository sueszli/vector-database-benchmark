import collections
import itertools

class UnionFind(object):

    def __init__(self):
        if False:
            for i in range(10):
                print('nop')
        self.set = {}
        self.rank = collections.Counter()

    def find_set(self, x):
        if False:
            return 10
        (xp, xr) = self.set.setdefault(x, (x, 1.0))
        if x != xp:
            (pp, pr) = self.find_set(xp)
            self.set[x] = (pp, xr * pr)
        return self.set[x]

    def union_set(self, x, y, r):
        if False:
            i = 10
            return i + 15
        ((xp, xr), (yp, yr)) = map(self.find_set, (x, y))
        if xp == yp:
            return False
        if self.rank[xp] < self.rank[yp]:
            self.set[xp] = (yp, r * yr / xr)
        elif self.rank[xp] > self.rank[yp]:
            self.set[yp] = (xp, 1.0 / r * xr / yr)
        else:
            self.set[yp] = (xp, 1.0 / r * xr / yr)
            self.rank[xp] += 1
        return True

    def query_set(self, x, y):
        if False:
            while True:
                i = 10
        if x not in self.set or y not in self.set:
            return -1.0
        ((xp, xr), (yp, yr)) = map(self.find_set, (x, y))
        return xr / yr if xp == yp else -1.0

class UnionFindPathCompressionOnly(object):

    def __init__(self):
        if False:
            return 10
        self.set = {}

    def find_set(self, x):
        if False:
            return 10
        (xp, xr) = self.set.setdefault(x, (x, 1.0))
        if x != xp:
            (pp, pr) = self.find_set(xp)
            self.set[x] = (pp, xr * pr)
        return self.set[x]

    def union_set(self, x, y, r):
        if False:
            i = 10
            return i + 15
        ((xp, xr), (yp, yr)) = map(self.find_set, (x, y))
        if xp == yp:
            return False
        self.set[xp] = (yp, r * yr / xr)
        return True

    def query_set(self, x, y):
        if False:
            print('Hello World!')
        if x not in self.set or y not in self.set:
            return -1.0
        ((xp, xr), (yp, yr)) = map(self.find_set, (x, y))
        return xr / yr if xp == yp else -1.0

class Solution(object):

    def calcEquation(self, equations, values, queries):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type equations: List[List[str]]\n        :type values: List[float]\n        :type queries: List[List[str]]\n        :rtype: List[float]\n        '
        union_find = UnionFind()
        for ((a, b), k) in itertools.izip(equations, values):
            union_find.union_set(a, b, k)
        return [union_find.query_set(a, b) for (a, b) in queries]
import collections
import itertools

class Solution2(object):

    def calcEquation(self, equations, values, queries):
        if False:
            i = 10
            return i + 15
        '\n        :type equations: List[List[str]]\n        :type values: List[float]\n        :type queries: List[List[str]]\n        :rtype: List[float]\n        '
        adj = collections.defaultdict(dict)
        for ((a, b), k) in itertools.izip(equations, values):
            adj[a][b] = k
            adj[b][a] = 1.0 / k

        def bfs(adj, a, b, lookup):
            if False:
                i = 10
                return i + 15
            if a not in adj or b not in adj:
                return -1.0
            if (a, b) in lookup:
                return lookup[a, b]
            visited = {a}
            q = collections.deque([(a, 1.0)])
            while q:
                (u, val) = q.popleft()
                if u == b:
                    lookup[a, b] = val
                    return val
                for (v, k) in adj[u].iteritems():
                    if v in visited:
                        continue
                    visited.add(v)
                    q.append((v, val * k))
            lookup[a, b] = -1.0
            return -1.0
        lookup = {}
        return [bfs(adj, a, b, lookup) for (a, b) in queries]
import collections
import itertools

class Solution3(object):

    def calcEquation(self, equations, values, queries):
        if False:
            return 10
        '\n        :type equations: List[List[str]]\n        :type values: List[float]\n        :type queries: List[List[str]]\n        :rtype: List[float]\n        '
        adj = collections.defaultdict(dict)
        for ((a, b), k) in itertools.izip(equations, values):
            adj[a][a] = adj[b][b] = 1.0
            adj[a][b] = k
            adj[b][a] = 1.0 / k
        for k in adj:
            for i in adj[k]:
                for j in adj[k]:
                    adj[i][j] = adj[i][k] * adj[k][j]
        return [adj[a].get(b, -1.0) for (a, b) in queries]
import collections

class Solution4(object):

    def calcEquation(self, equations, values, query):
        if False:
            while True:
                i = 10
        '\n        :type equations: List[List[str]]\n        :type values: List[float]\n        :type query: List[List[str]]\n        :rtype: List[float]\n        '

        def check(up, down, lookup, visited):
            if False:
                return 10
            if up in lookup and down in lookup[up]:
                return (True, lookup[up][down])
            for (k, v) in lookup[up].iteritems():
                if k not in visited:
                    visited.add(k)
                    tmp = check(k, down, lookup, visited)
                    if tmp[0]:
                        return (True, v * tmp[1])
            return (False, 0)
        lookup = collections.defaultdict(dict)
        for (i, e) in enumerate(equations):
            lookup[e[0]][e[1]] = values[i]
            if values[i]:
                lookup[e[1]][e[0]] = 1.0 / values[i]
        result = []
        for q in query:
            visited = set()
            tmp = check(q[0], q[1], lookup, visited)
            result.append(tmp[1] if tmp[0] else -1)
        return result