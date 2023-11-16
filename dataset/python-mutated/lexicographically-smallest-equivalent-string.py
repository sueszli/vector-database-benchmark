class UnionFind(object):

    def __init__(self, n):
        if False:
            while True:
                i = 10
        self.set = range(n)

    def find_set(self, x):
        if False:
            while True:
                i = 10
        if self.set[x] != x:
            self.set[x] = self.find_set(self.set[x])
        return self.set[x]

    def union_set(self, x, y):
        if False:
            for i in range(10):
                print('nop')
        (x_root, y_root) = map(self.find_set, (x, y))
        if x_root == y_root:
            return False
        self.set[max(x_root, y_root)] = min(x_root, y_root)
        return True

class Solution(object):

    def smallestEquivalentString(self, A, B, S):
        if False:
            i = 10
            return i + 15
        '\n        :type A: str\n        :type B: str\n        :type S: str\n        :rtype: str\n        '
        union_find = UnionFind(26)
        for i in xrange(len(A)):
            union_find.union_set(ord(A[i]) - ord('a'), ord(B[i]) - ord('a'))
        result = []
        for i in xrange(len(S)):
            parent = union_find.find_set(ord(S[i]) - ord('a'))
            result.append(chr(parent + ord('a')))
        return ''.join(result)