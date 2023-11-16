import collections

class Solution(object):

    def diagonalSort(self, mat):
        if False:
            while True:
                i = 10
        '\n        :type mat: List[List[int]]\n        :rtype: List[List[int]]\n        '
        lookup = collections.defaultdict(list)
        for i in xrange(len(mat)):
            for j in xrange(len(mat[0])):
                lookup[i - j].append(mat[i][j])
        for v in lookup.itervalues():
            v.sort()
        for i in reversed(xrange(len(mat))):
            for j in reversed(xrange(len(mat[0]))):
                mat[i][j] = lookup[i - j].pop()
        return mat