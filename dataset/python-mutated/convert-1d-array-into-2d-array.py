class Solution(object):

    def construct2DArray(self, original, m, n):
        if False:
            i = 10
            return i + 15
        '\n        :type original: List[int]\n        :type m: int\n        :type n: int\n        :rtype: List[List[int]]\n        '
        return [original[i:i + n] for i in xrange(0, len(original), n)] if len(original) == m * n else []