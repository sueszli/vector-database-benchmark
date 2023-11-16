class Solution(object):

    def minSkips(self, dist, speed, hoursBefore):
        if False:
            while True:
                i = 10
        '\n        :type dist: List[int]\n        :type speed: int\n        :type hoursBefore: int\n        :rtype: int\n        '

        def ceil(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b - 1) // b
        dp = [0] * (len(dist) - 1 + 1)
        for (i, d) in enumerate(dist):
            for j in reversed(xrange(len(dp))):
                dp[j] = ceil(dp[j] + d, speed) * speed if i < len(dist) - 1 else dp[j] + d
                if j - 1 >= 0:
                    dp[j] = min(dp[j], dp[j - 1] + d)
        target = hoursBefore * speed
        for i in xrange(len(dist)):
            if dp[i] <= target:
                return i
        return -1