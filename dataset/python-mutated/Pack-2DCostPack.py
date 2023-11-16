class Solution:

    def twoDCostPackMethod1(self, weight: [int], volume: [int], value: [int], W: int, V: int):
        if False:
            for i in range(10):
                print('nop')
        size = len(weight)
        dp = [[[0 for _ in range(V + 1)] for _ in range(W + 1)] for _ in range(size + 1)]
        for i in range(1, N + 1):
            for w in range(W + 1):
                for v in range(V + 1):
                    if w < weight[i - 1] or v < volume[i - 1]:
                        dp[i][w][v] = dp[i - 1][w][v]
                    else:
                        dp[i][w][v] = max(dp[i - 1][w][v], dp[i - 1][w - weight[i - 1]][v - volume[i - 1]] + value[i - 1])
        return dp[size][W][V]

    def twoDCostPackMethod2(self, weight: [int], volume: [int], value: [int], W: int, V: int):
        if False:
            i = 10
            return i + 15
        size = len(weight)
        dp = [[0 for _ in range(V + 1)] for _ in range(W + 1)]
        for i in range(1, N + 1):
            for w in range(W, weight[i - 1] - 1, -1):
                for v in range(V, volume[i - 1] - 1, -1):
                    dp[w][v] = max(dp[w][v], dp[w - weight[i - 1]][v - volume[i - 1]] + value[i - 1])
        return dp[W][V]