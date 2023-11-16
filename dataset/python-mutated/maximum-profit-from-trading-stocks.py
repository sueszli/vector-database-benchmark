import itertools

class Solution(object):

    def maximumProfit(self, present, future, budget):
        if False:
            print('Hello World!')
        '\n        :type present: List[int]\n        :type future: List[int]\n        :type budget: int\n        :rtype: int\n        '
        dp = [0] * (budget + 1)
        for (i, (p, f)) in enumerate(itertools.izip(present, future)):
            if f - p < 0:
                continue
            for b in reversed(xrange(p, budget + 1)):
                dp[b] = max(dp[b], dp[b - p] + (f - p))
        return dp[-1]
import itertools

class Solution2(object):

    def maximumProfit(self, present, future, budget):
        if False:
            print('Hello World!')
        '\n        :type present: List[int]\n        :type future: List[int]\n        :type budget: int\n        :rtype: int\n        '
        dp = [[0] * (budget + 1) for _ in xrange(2)]
        for (i, (p, f)) in enumerate(itertools.izip(present, future)):
            for b in xrange(budget + 1):
                dp[(i + 1) % 2][b] = max(dp[i % 2][b], dp[i % 2][b - p] + (f - p) if b - p >= 0 else 0)
        return dp[len(present) % 2][-1]