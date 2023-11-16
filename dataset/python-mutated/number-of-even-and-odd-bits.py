class Solution(object):

    def evenOddBit(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: List[int]\n        '

        def popcount(x):
            if False:
                return 10
            return bin(x)[2:].count('1')
        return [popcount(n & 341), popcount(n & 682)]