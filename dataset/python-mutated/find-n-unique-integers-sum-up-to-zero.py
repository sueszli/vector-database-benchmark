class Solution(object):

    def sumZero(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: List[int]\n        '
        return [i for i in xrange(-(n // 2), n // 2 + 1) if not (i == 0 and n % 2 == 0)]