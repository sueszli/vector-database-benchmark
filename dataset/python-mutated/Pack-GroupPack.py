class Solution:

    def groupPackMethod1(self, group_count: [int], weight: [[int]], value: [[int]], W: int):
        if False:
            i = 10
            return i + 15
        size = len(group_count)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]
        for i in range(1, size + 1):
            for w in range(W + 1):
                dp[i][w] = dp[i - 1][w]
                for k in range(group_count[i - 1]):
                    if w >= weight[i - 1][k]:
                        dp[i][w] = max(dp[i][w], dp[i - 1][w - weight[i - 1][k]] + value[i - 1][k])
        return dp[size][W]

    def groupPackMethod2(self, group_count: [int], weight: [[int]], value: [[int]], W: int):
        if False:
            i = 10
            return i + 15
        size = len(group_count)
        dp = [0 for _ in range(W + 1)]
        for i in range(1, size + 1):
            for w in range(W, -1, -1):
                for k in range(group_count[i - 1]):
                    if w >= weight[i - 1][k]:
                        dp[w] = max(dp[w], dp[w - weight[i - 1][k]] + value[i - 1][k])
        return dp[W]