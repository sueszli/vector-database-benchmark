class Solution(object):

    def minimumFinishTime(self, tires, changeTime, numLaps):
        if False:
            return 10
        '\n        :type tires: List[List[int]]\n        :type changeTime: int\n        :type numLaps: int\n        :rtype: int\n        '

        def ceil_log2(x):
            if False:
                return 10
            return (x - 1).bit_length()
        dp = [float('inf')] * ceil_log2(changeTime + 1)
        for (f, r) in tires:
            total = curr = f
            cnt = 0
            while curr < changeTime + f:
                dp[cnt] = min(dp[cnt], total)
                curr *= r
                total += curr
                cnt += 1
        dp2 = [float('inf')] * numLaps
        for i in xrange(numLaps):
            dp2[i] = min(((dp2[i - j - 1] + changeTime if i - j - 1 >= 0 else 0) + dp[j] for j in xrange(min(i + 1, len(dp)))))
        return dp2[-1]