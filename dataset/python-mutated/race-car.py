class Solution(object):

    def racecar(self, target):
        if False:
            while True:
                i = 10
        dp = [0] * (target + 1)
        for i in xrange(1, target + 1):
            k = i.bit_length()
            if i == 2 ** k - 1:
                dp[i] = k
                continue
            dp[i] = k + 1 + dp[2 ** k - 1 - i]
            for j in xrange(k - 1):
                dp[i] = min(dp[i], k + j + 1 + dp[i - 2 ** (k - 1) + 2 ** j])
        return dp[-1]