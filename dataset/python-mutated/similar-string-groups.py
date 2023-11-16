import collections
import itertools

class UnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.set = range(n)
        self.__size = n

    def find_set(self, x):
        if False:
            print('Hello World!')
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            i = 10
            return i + 15
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[min(x_root, y_root)] = max(x_root, y_root)
        self.__size -= 1
        return True

    def size(self):
        if False:
            print('Hello World!')
        return self.__size

class Solution(object):

    def numSimilarGroups(self, A):
        if False:
            i = 10
            return i + 15

        def isSimilar(a, b):
            if False:
                print('Hello World!')
            diff = 0
            for (x, y) in itertools.izip(a, b):
                if x != y:
                    diff += 1
                    if diff > 2:
                        return False
            return diff == 2
        (N, L) = (len(A), len(A[0]))
        union_find = UnionFind(N)
        if N < L * L:
            for ((i1, word1), (i2, word2)) in itertools.combinations(enumerate(A), 2):
                if isSimilar(word1, word2):
                    union_find.union_set(i1, i2)
        else:
            buckets = collections.defaultdict(list)
            lookup = set()
            for i in xrange(len(A)):
                word = list(A[i])
                if A[i] not in lookup:
                    buckets[A[i]].append(i)
                    lookup.add(A[i])
                for (j1, j2) in itertools.combinations(xrange(L), 2):
                    (word[j1], word[j2]) = (word[j2], word[j1])
                    buckets[''.join(word)].append(i)
                    (word[j1], word[j2]) = (word[j2], word[j1])
            for word in A:
                for (i1, i2) in itertools.combinations(buckets[word], 2):
                    union_find.union_set(i1, i2)
        return union_find.size()