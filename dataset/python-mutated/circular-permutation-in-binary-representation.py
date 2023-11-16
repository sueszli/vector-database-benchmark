class Solution(object):

    def circularPermutation(self, n, start):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :type start: int\n        :rtype: List[int]\n        '
        return [start ^ i >> 1 ^ i for i in xrange(1 << n)]