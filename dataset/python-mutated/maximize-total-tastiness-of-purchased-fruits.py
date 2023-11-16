import itertools

class Solution(object):

    def maxTastiness(self, price, tastiness, maxAmount, maxCoupons):
        if False:
            return 10
        '\n        :type price: List[int]\n        :type tastiness: List[int]\n        :type maxAmount: int\n        :type maxCoupons: int\n        :rtype: int\n        '
        dp = [[0] * (maxCoupons + 1) for _ in xrange(maxAmount + 1)]
        for (p, t) in itertools.izip(price, tastiness):
            for i in reversed(xrange(p // 2, maxAmount + 1)):
                for j in reversed(xrange(maxCoupons + 1)):
                    if i - p >= 0:
                        dp[i][j] = max(dp[i][j], t + dp[i - p][j])
                    if j - 1 >= 0:
                        dp[i][j] = max(dp[i][j], t + dp[i - p // 2][j - 1])
        return dp[maxAmount][maxCoupons]