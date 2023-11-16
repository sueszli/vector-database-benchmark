class UnionFind(object):

    def __init__(self, n):
        if False:
            i = 10
            return i + 15
        self.set = range(n)
        self.rank = [0] * n
        self.size = [1] * n
        self.total = n

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
            for i in range(10):
                print('nop')
        (x, y) = (self.find_set(x), self.find_set(y))
        if x == y:
            return False
        if self.rank[x] > self.rank[y]:
            (x, y) = (y, x)
        self.set[x] = self.set[y]
        if self.rank[x] == self.rank[y]:
            self.rank[y] += 1
        self.size[y] += self.size[x]
        self.total -= 1
        return True

class Solution(object):

    def groupStrings(self, words):
        if False:
            print('Hello World!')
        '\n        :type words: List[str]\n        :rtype: List[int]\n        '
        uf = UnionFind(len(words))
        lookup = {}
        for (i, x) in enumerate(words):
            mask = reduce(lambda x, y: x | 1 << ord(y) - ord('a'), x, 0)
            if mask not in lookup:
                lookup[mask] = i
            uf.union_set(i, lookup[mask])
            bit = 1
            while bit <= mask:
                if mask & bit:
                    if mask ^ bit not in lookup:
                        lookup[mask ^ bit] = i
                    uf.union_set(i, lookup[mask ^ bit])
                bit <<= 1
        return [uf.total, max(uf.size)]