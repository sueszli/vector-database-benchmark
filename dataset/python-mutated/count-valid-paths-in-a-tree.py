class Solution(object):

    def countPaths(self, n, edges):
        if False:
            return 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def linear_sieve_of_eratosthenes(n):
            if False:
                return 10
            primes = []
            spf = [-1] * (n + 1)
            for i in xrange(2, n + 1):
                if spf[i] == -1:
                    spf[i] = i
                    primes.append(i)
                for p in primes:
                    if i * p > n or p > spf[i]:
                        break
                    spf[i * p] = p
            return spf

        def is_prime(u):
            if False:
                i = 10
                return i + 15
            return spf[u] == u

        def iter_dfs():
            if False:
                for i in range(10):
                    print('nop')
            result = 0
            stk = [(1, (0, -1, [0] * 2))]
            while stk:
                (step, args) = stk.pop()
                if step == 1:
                    (u, p, ret) = args
                    ret[:] = [1 - is_prime(u + 1), is_prime(u + 1)]
                    stk.append((2, (u, p, ret, 0)))
                elif step == 2:
                    (u, p, ret, i) = args
                    if i == len(adj[u]):
                        continue
                    v = adj[u][i]
                    if v == p:
                        stk.append((2, (u, p, ret, i + 1)))
                        continue
                    new_ret = [0] * 2
                    stk.append((3, (u, p, new_ret, ret, i)))
                    stk.append((1, (v, u, new_ret)))
                elif step == 3:
                    (u, p, new_ret, ret, i) = args
                    result += ret[0] * new_ret[1] + ret[1] * new_ret[0]
                    if is_prime(u + 1):
                        ret[1] += new_ret[0]
                    else:
                        ret[0] += new_ret[0]
                        ret[1] += new_ret[1]
                    stk.append((2, (u, p, ret, i + 1)))
            return result
        spf = linear_sieve_of_eratosthenes(n)
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            (u, v) = (u - 1, v - 1)
            adj[u].append(v)
            adj[v].append(u)
        return iter_dfs()

class Solution2(object):

    def countPaths(self, n, edges):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def linear_sieve_of_eratosthenes(n):
            if False:
                i = 10
                return i + 15
            primes = []
            spf = [-1] * (n + 1)
            for i in xrange(2, n + 1):
                if spf[i] == -1:
                    spf[i] = i
                    primes.append(i)
                for p in primes:
                    if i * p > n or p > spf[i]:
                        break
                    spf[i * p] = p
            return spf

        def is_prime(u):
            if False:
                while True:
                    i = 10
            return spf[u] == u

        def dfs(u, p):
            if False:
                return 10
            cnt = [1 - is_prime(u + 1), is_prime(u + 1)]
            for v in adj[u]:
                if v == p:
                    continue
                new_cnt = dfs(v, u)
                result[0] += cnt[0] * new_cnt[1] + cnt[1] * new_cnt[0]
                if is_prime(u + 1):
                    cnt[1] += new_cnt[0]
                else:
                    cnt[0] += new_cnt[0]
                    cnt[1] += new_cnt[1]
            return cnt
        spf = linear_sieve_of_eratosthenes(n)
        adj = [[] for _ in xrange(n)]
        for (u, v) in edges:
            (u, v) = (u - 1, v - 1)
            adj[u].append(v)
            adj[v].append(u)
        result = [0]
        dfs(0, -1)
        return result[0]

class UnionFind(object):

    def __init__(self, n):
        if False:
            return 10
        self.set = range(n)
        self.rank = [0] * n
        self.size = [1] * n

    def find_set(self, x):
        if False:
            return 10
        stk = []
        while self.set[x] != x:
            stk.append(x)
            x = self.set[x]
        while stk:
            self.set[stk.pop()] = x
        return x

    def union_set(self, x, y):
        if False:
            print('Hello World!')
        (x, y) = (self.find_set(x), self.find_set(y))
        if x == y:
            return False
        if self.rank[x] > self.rank[y]:
            (x, y) = (y, x)
        self.set[x] = self.set[y]
        if self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        self.size[y] += self.size[x]
        return True

    def total(self, x):
        if False:
            for i in range(10):
                print('nop')
        return self.size[self.find_set(x)]

class Solution3(object):

    def countPaths(self, n, edges):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type n: int\n        :type edges: List[List[int]]\n        :rtype: int\n        '

        def linear_sieve_of_eratosthenes(n):
            if False:
                for i in range(10):
                    print('nop')
            primes = []
            spf = [-1] * (n + 1)
            for i in xrange(2, n + 1):
                if spf[i] == -1:
                    spf[i] = i
                    primes.append(i)
                for p in primes:
                    if i * p > n or p > spf[i]:
                        break
                    spf[i * p] = p
            return spf

        def is_prime(u):
            if False:
                while True:
                    i = 10
            return spf[u] == u
        spf = linear_sieve_of_eratosthenes(n)
        uf = UnionFind(n)
        for (u, v) in edges:
            (u, v) = (u - 1, v - 1)
            if is_prime(u + 1) == is_prime(v + 1) == False:
                uf.union_set(u, v)
        result = 0
        cnt = [1] * n
        for (u, v) in edges:
            (u, v) = (u - 1, v - 1)
            if is_prime(u + 1) == is_prime(v + 1):
                continue
            if not is_prime(u + 1):
                (u, v) = (v, u)
            result += cnt[u] * uf.total(v)
            cnt[u] += uf.total(v)
        return result