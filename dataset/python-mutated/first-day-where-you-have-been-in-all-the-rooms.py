class Solution(object):

    def firstDayBeenInAllRooms(self, nextVisit):
        if False:
            i = 10
            return i + 15
        '\n        :type nextVisit: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        dp = [0] * len(nextVisit)
        for i in xrange(1, len(dp)):
            dp[i] = (dp[i - 1] + 1 + (dp[i - 1] - dp[nextVisit[i - 1]]) + 1) % MOD
        return dp[-1]