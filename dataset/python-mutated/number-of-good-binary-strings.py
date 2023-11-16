class Solution(object):

    def goodBinaryStrings(self, minLength, maxLength, oneGroup, zeroGroup):
        if False:
            while True:
                i = 10
        '\n        :type minLength: int\n        :type maxLength: int\n        :type oneGroup: int\n        :type zeroGroup: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        result = 0
        w = max(oneGroup, zeroGroup) + 1
        dp = [0] * w
        for i in xrange(maxLength + 1):
            dp[i % w] = 1 if i == 0 else 0
            if i - oneGroup >= 0:
                dp[i % w] = (dp[i % w] + dp[(i - oneGroup) % w]) % MOD
            if i - zeroGroup >= 0:
                dp[i % w] = (dp[i % w] + dp[(i - zeroGroup) % w]) % MOD
            if i >= minLength:
                result = (result + dp[i % w]) % MOD
        return result

class Solution(object):

    def goodBinaryStrings(self, minLength, maxLength, oneGroup, zeroGroup):
        if False:
            i = 10
            return i + 15
        '\n        :type minLength: int\n        :type maxLength: int\n        :type oneGroup: int\n        :type zeroGroup: int\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        result = 0
        w = max(oneGroup, zeroGroup) + 1
        dp = [0] * w
        dp[0] = 1
        for i in xrange(maxLength + 1):
            if i >= minLength:
                result = (result + dp[i % w]) % MOD
            if i + oneGroup <= maxLength:
                dp[(i + oneGroup) % w] = (dp[(i + oneGroup) % w] + dp[i % w]) % MOD
            if i + zeroGroup <= maxLength:
                dp[(i + zeroGroup) % w] = (dp[(i + zeroGroup) % w] + dp[i % w]) % MOD
            dp[i % w] = 0
        return result