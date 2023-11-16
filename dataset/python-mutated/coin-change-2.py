class Solution(object):

    def change(self, amount, coins):
        if False:
            i = 10
            return i + 15
        '\n        :type amount: int\n        :type coins: List[int]\n        :rtype: int\n        '
        dp = [0] * (amount + 1)
        dp[0] = 1
        for coin in coins:
            for i in xrange(coin, amount + 1):
                dp[i] += dp[i - coin]
        return dp[amount]