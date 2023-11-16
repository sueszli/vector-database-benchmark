class Solution:

    def mixedPackMethod1(self, weight: [int], value: [int], count: [int], W: int):
        if False:
            return 10
        (weight_new, value_new, count_new) = ([], [], [])
        for i in range(len(weight)):
            cnt = count[i]
            if cnt > 0:
                k = 1
                while k <= cnt:
                    cnt -= k
                    weight_new.append(weight[i] * k)
                    value_new.append(value[i] * k)
                    count_new.append(1)
                    k *= 2
                if cnt > 0:
                    weight_new.append(weight[i] * cnt)
                    value_new.append(value[i] * cnt)
                    count_new.append(1)
            elif cnt == -1:
                weight_new.append(weight[i])
                value_new.append(value[i])
                count_new.append(1)
            else:
                weight_new.append(weight[i])
                value_new.append(value[i])
                count_new.append(0)
        dp = [0 for _ in range(W + 1)]
        size = len(weight_new)
        for i in range(1, size + 1):
            if count_new[i - 1] == 1:
                for w in range(W, weight_new[i - 1] - 1, -1):
                    dp[w] = max(dp[w], dp[w - weight_new[i - 1]] + value_new[i - 1])
            else:
                for w in range(weight_new[i - 1], W + 1):
                    dp[w] = max(dp[w], dp[w - weight_new[i - 1]] + value_new[i - 1])
        return dp[W]