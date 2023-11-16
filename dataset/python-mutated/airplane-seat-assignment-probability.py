class Solution(object):

    def nthPersonGetsNthSeat(self, n):
        if False:
            return 10
        '\n        :type n: int\n        :rtype: float\n        '
        return 0.5 if n != 1 else 1.0

class Solution2(object):

    def nthPersonGetsNthSeat(self, n):
        if False:
            print('Hello World!')
        '\n        :type n: int\n        :rtype: float\n        '
        dp = [0.0] * 2
        dp[0] = 1.0
        for i in xrange(2, n + 1):
            dp[(i - 1) % 2] = 1.0 / i + dp[(i - 2) % 2] * (i - 2) / i
        return dp[(n - 1) % 2]