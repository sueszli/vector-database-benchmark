class Solution:

    def zeroOnePackMethod1(self, weight: [int], value: [int], W: int):
        if False:
            while True:
                i = 10
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]
        for i in range(1, size + 1):
            for w in range(W + 1):
                if w < weight[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                else:
                    dp[i][w] = max(dp[i - 1][w], dp[i - 1][w - weight[i - 1]] + value[i - 1])
        return dp[size][W]

    def zeroOnePackMethod2(self, weight: [int], value: [int], W: int):
        if False:
            while True:
                i = 10
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        for i in range(1, size + 1):
            for w in range(W, weight[i - 1] - 1, -1):
                dp[w] = max(dp[w], dp[w - weight[i - 1]] + value[i - 1])
        return dp[W]