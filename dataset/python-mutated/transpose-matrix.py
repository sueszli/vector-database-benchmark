class Solution(object):

    def transpose(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[List[int]]\n        :rtype: List[List[int]]\n        '
        result = [[None] * len(A) for _ in xrange(len(A[0]))]
        for (r, row) in enumerate(A):
            for (c, val) in enumerate(row):
                result[c][r] = val
        return result

class Solution2(object):

    def transpose(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[List[int]]\n        :rtype: List[List[int]]\n        '
        return zip(*A)