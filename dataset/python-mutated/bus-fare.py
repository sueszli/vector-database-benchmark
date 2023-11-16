class Solution:

    def solve(self, days):
        if False:
            i = 10
            return i + 15
        prices = [2, 7, 25]
        durations = [1, 7, 30]
        n = len(days)
        m = len(prices)
        dp = [float('inf')] * (n + 1)
        dp[0] = 0
        pointers = [0] * m
        for i in range(1, n + 1):
            for j in range(m):
                while days[i - 1] - days[pointers[j]] >= durations[j]:
                    pointers[j] += 1
                dp[i] = min(dp[i], dp[pointers[j]] + prices[j])
        return dp[-1]