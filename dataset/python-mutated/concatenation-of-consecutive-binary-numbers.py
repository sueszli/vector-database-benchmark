class Solution(object):

    def concatenatedBinary(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        result = l = 0
        for i in xrange(1, n + 1):
            if i & i - 1 == 0:
                l += 1
            result = ((result << l) % MOD + i) % MOD
        return result