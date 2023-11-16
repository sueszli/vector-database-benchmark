class Solution:

    def zeroOnePackJustFillUp(self, weight: [int], value: [int], W: int):
        if False:
            for i in range(10):
                print('nop')
        size = len(weight)
        dp = [float('-inf') for _ in range(W + 1)]
        dp[0] = 0
        for i in range(1, size + 1):
            for w in range(W, weight[i - 1] - 1, -1):
                dp[w] = max(dp[w], dp[w - weight[i - 1]] + value[i - 1])
        if dp[W] == float('-inf'):
            return -1
        return dp[W]

    def completePackJustFillUp(self, weight: [int], value: [int], W: int):
        if False:
            return 10
        size = len(weight)
        dp = [float('-inf') for _ in range(W + 1)]
        dp[0] = 0
        for i in range(1, size + 1):
            for w in range(weight[i - 1], W + 1):
                dp[w] = max(dp[w], dp[w - weight[i - 1]] + value[i - 1])
        if dp[W] == float('-inf'):
            return -1
        return dp[W]

    def zeroOnePackNumbers(self, weight: [int], value: [int], W: int):
        if False:
            while True:
                i = 10
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        dp[0] = 1
        for i in range(1, size + 1):
            for w in range(W, weight[i - 1] - 1, -1):
                dp[w] = dp[w] + dp[w - weight[i - 1]]
        return dp[W]

    def completePackNumbers(self, weight: [int], value: [int], W: int):
        if False:
            for i in range(10):
                print('nop')
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        dp[0] = 1
        for i in range(1, size + 1):
            for w in range(weight[i - 1], W + 1):
                dp[w] = dp[w] + dp[w - weight[i - 1]]
        return dp[W]

    def zeroOnePackMaxProfitNumbers1(self, weight: [int], value: [int], W: int):
        if False:
            print('Hello World!')
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]
        op = [[1 for _ in range(W + 1)] for _ in range(size + 1)]
        for i in range(1, size + 1):
            for w in range(W + 1):
                if w < weight[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    op[i][w] = op[i - 1][w]
                elif dp[i - 1][w] < dp[i - 1][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i - 1][w - weight[i - 1]] + value[i - 1]
                    op[i][w] = op[i - 1][w - weight[i - 1]]
                elif dp[i - 1][w] == dp[i - 1][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    op[i][w] = op[i - 1][w] + op[i - 1][w - weight[i - 1]]
                else:
                    dp[i][w] = dp[i - 1][w]
                    op[i][w] = op[i - 1][w]
        return op[size][W]

    def zeroOnePackMaxProfitNumbers2(self, weight: [int], value: [int], W: int):
        if False:
            i = 10
            return i + 15
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        op = [1 for _ in range(W + 1)]
        for i in range(1, size + 1):
            for w in range(W, weight[i - 1] - 1, -1):
                if dp[w] < dp[w - weight[i - 1]] + value[i - 1]:
                    dp[w] = dp[w - weight[i - 1]] + value[i - 1]
                    op[w] = op[w - weight[i - 1]]
                elif dp[w] == dp[w - weight[i - 1]] + value[i - 1]:
                    op[w] = op[w] + op[w - weight[i - 1]]
        return op[W]

    def completePackMaxProfitNumbers1(self, weight: [int], value: [int], W: int):
        if False:
            for i in range(10):
                print('nop')
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]
        op = [[1 for _ in range(W + 1)] for _ in range(size + 1)]
        for i in range(1, size + 1):
            for w in range(W + 1):
                if w < weight[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    op[i][w] = op[i - 1][w]
                elif dp[i - 1][w] < dp[i][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i][w - weight[i - 1]] + value[i - 1]
                    op[i][w] = op[i][w - weight[i - 1]]
                elif dp[i - 1][w] == dp[i][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    op[i][w] = op[i - 1][w] + op[i][w - weight[i - 1]]
                else:
                    dp[i][w] = dp[i - 1][w]
                    op[i][w] = op[i - 1][w]
        return dp[size][W]

    def completePackMaxProfitNumbers2(self, weight: [int], value: [int], W: int):
        if False:
            return 10
        size = len(weight)
        dp = [0 for _ in range(W + 1)]
        op = [1 for _ in range(W + 1)]
        for i in range(1, size + 1):
            for w in range(weight[i - 1], W + 1):
                if dp[w] < dp[w - weight[i - 1]] + value[i - 1]:
                    dp[w] = dp[w - weight[i - 1]] + value[i - 1]
                    op[w] = op[w - weight[i - 1]]
                elif dp[w] == dp[w - weight[i - 1]] + value[i - 1]:
                    op[w] = op[w] + op[w - weight[i - 1]]
        return dp[size][W]

    def zeroOnePackPrintPath(self, weight: [int], value: [int], W: int):
        if False:
            return 10
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]
        path = [[False for _ in range(W + 1)] for _ in range(size + 1)]
        for i in range(1, size + 1):
            for w in range(W + 1):
                if w < weight[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    path[i][w] = False
                elif dp[i - 1][w] < dp[i - 1][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i - 1][w - weight[i - 1]] + value[i - 1]
                    path[i][w] = True
                elif dp[i - 1][w] == dp[i - 1][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    path[i][w] = True
                else:
                    dp[i][w] = dp[i - 1][w]
                    path[i][w] = False
        res = []
        (i, w) = (size, W)
        while i >= 1 and w >= 0:
            if path[i][w]:
                res.append(str(i - 1))
                w -= weight[i - 1]
            i -= 1
        return ' '.join(res[::-1])

    def zeroOnePackPrintPathMinOrder(self, weight: [int], value: [int], W: int):
        if False:
            print('Hello World!')
        size = len(weight)
        dp = [[0 for _ in range(W + 1)] for _ in range(size + 1)]
        path = [[False for _ in range(W + 1)] for _ in range(size + 1)]
        weight.reverse()
        value.reverse()
        for i in range(1, size + 1):
            for w in range(W + 1):
                if w < weight[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    path[i][w] = False
                elif dp[i - 1][w] < dp[i - 1][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i - 1][w - weight[i - 1]] + value[i - 1]
                    path[i][w] = True
                elif dp[i - 1][w] == dp[i - 1][w - weight[i - 1]] + value[i - 1]:
                    dp[i][w] = dp[i - 1][w]
                    path[i][w] = True
                else:
                    dp[i][w] = dp[i - 1][w]
                    path[i][w] = False
        res = []
        (i, w) = (size, W)
        while i >= 1 and w >= 0:
            if path[i][w]:
                res.append(str(size - i))
                w -= weight[i - 1]
            i -= 1
        return ' '.join(res)