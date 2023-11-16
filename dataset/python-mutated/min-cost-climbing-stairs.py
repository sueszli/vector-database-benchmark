class Solution(object):

    def minCostClimbingStairs(self, cost):
        if False:
            i = 10
            return i + 15
        '\n        :type cost: List[int]\n        :rtype: int\n        '
        dp = [0] * 3
        for i in reversed(xrange(len(cost))):
            dp[i % 3] = cost[i] + min(dp[(i + 1) % 3], dp[(i + 2) % 3])
        return min(dp[0], dp[1])