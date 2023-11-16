class Solution(object):

    def smallestCommonElement(self, mat):
        if False:
            while True:
                i = 10
        '\n        :type mat: List[List[int]]\n        :rtype: int\n        '
        intersections = set(mat[0])
        for i in xrange(1, len(mat)):
            intersections &= set(mat[i])
            if not intersections:
                return -1
        return min(intersections)
import collections

class Solution2(object):

    def smallestCommonElement(self, mat):
        if False:
            return 10
        '\n        :type mat: List[List[int]]\n        :rtype: int\n        '
        counter = collections.Counter()
        for row in mat:
            for c in row:
                counter[c] += 1
                if counter[c] == len(mat):
                    return c
        return -1