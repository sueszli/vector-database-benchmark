class Solution(object):

    def minimumTime(self, power):
        if False:
            while True:
                i = 10
        '\n        :type power: List[int]\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                return 10
            return (a + b - 1) // b
        INF = float('inf')
        dp = {0: 0}
        for gain in xrange(1, len(power) + 1):
            new_dp = collections.defaultdict(lambda : INF)
            for mask in dp.iterkeys():
                for i in xrange(len(power)):
                    if mask & 1 << i == 0:
                        new_dp[mask | 1 << i] = min(new_dp[mask | 1 << i], dp[mask] + ceil_divide(power[i], gain))
            dp = new_dp
        return dp[(1 << len(power)) - 1]