class Solution:

    def multiplePackMethod1(self, weight: [int], value: [int], count: [int], W: int):
        if False:
            print('Hello World!')
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]
        for i in range(1, size + 1):
            for w in range(W + 1):
                for k in range(min(count[i - 1], w // weight[i - 1]) + 1):
                    dp[i][w] = max(dp[i][w], dp[i - 1][w - k * weight[i - 1]] + k * value[i - 1])
        return dp[size][W]

    def multiplePackMethod2(self, weight: [int], value: [int], count: [int], W: int):
        if False:
            while True:
                i = 10
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        for i in range(1, size + 1):
            for w in range(W, weight[i - 1] - 1, -1):
                for k in range(min(count[i - 1], w // weight[i - 1]) + 1):
                    dp[w] = max(dp[w], dp[w - k * weight[i - 1]] + k * value[i - 1])
        return dp[W]

    def multiplePackMethod3(self, weight: [int], value: [int], count: [int], W: int):
        if False:
            print('Hello World!')
        (weight_new, value_new) = ([], [])
        for i in range(len(weight)):
            cnt = count[i]
            k = 1
            while k <= cnt:
                cnt -= k
                weight_new.append(weight[i] * k)
                value_new.append(value[i] * k)
                k *= 2
            if cnt > 0:
                weight_new.append(weight[i] * cnt)
                value_new.append(value[i] * cnt)
        dp = [0 for _ in range(W + 1)]
        size = len(weight_new)
        for i in range(1, size + 1):
            for w in range(W, weight_new[i - 1] - 1, -1):
                dp[w] = max(dp[w], dp[w - weight_new[i - 1]] + value_new[i - 1])
        return dp[W]