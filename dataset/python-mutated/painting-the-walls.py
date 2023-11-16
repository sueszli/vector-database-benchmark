import itertools

class Solution(object):

    def paintWalls(self, cost, time):
        if False:
            return 10
        '\n        :type cost: List[int]\n        :type time: List[int]\n        :rtype: int\n        '
        dp = [float('inf')] * (len(cost) + 1)
        dp[0] = 0
        for (c, t) in itertools.izip(cost, time):
            for j in reversed(xrange(1, len(cost) + 1)):
                dp[j] = min(dp[j], dp[max(j - (t + 1), 0)] + c)
        return dp[-1]