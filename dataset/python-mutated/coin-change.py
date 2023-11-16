class Solution(object):

    def coinChange(self, coins, amount):
        if False:
            return 10
        '\n        :type coins: List[int]\n        :type amount: int\n        :rtype: int\n        '
        INF = 2147483647
        dp = [INF] * (amount + 1)
        dp[0] = 0
        for i in xrange(amount + 1):
            if dp[i] != INF:
                for coin in coins:
                    if i + coin <= amount:
                        dp[i + coin] = min(dp[i + coin], dp[i] + 1)
        return dp[amount] if dp[amount] != INF else -1