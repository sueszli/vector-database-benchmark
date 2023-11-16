import itertools

class UnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.set = range(n)
        self.rank = [0] * n

    def find_set(self, x):
        if False:
            i = 10
            return i + 15
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

    def gcdSort(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: bool\n        '

        def modified_sieve_of_eratosthenes(n, lookup, uf):
            if False:
                return 10
            if n < 2:
                return
            is_prime = [True] * (n + 1)
            for i in xrange(2, len(is_prime)):
                if not is_prime[i]:
                    continue
                for j in xrange(i + i, len(is_prime), i):
                    is_prime[j] = False
                    if j in lookup:
                        uf.union_set(i - 1, j - 1)
        max_num = max(nums)
        uf = UnionFind(max_num)
        modified_sieve_of_eratosthenes(max_num, set(nums), uf)
        return all((uf.find_set(a - 1) == uf.find_set(b - 1) for (a, b) in itertools.izip(nums, sorted(nums))))