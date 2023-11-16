class Solution(object):

    def maxSizeSlices(self, slices):
        if False:
            return 10
        '\n        :type slices: List[int]\n        :rtype: int\n        '

        def maxSizeSlicesLinear(slices, start, end):
            if False:
                while True:
                    i = 10
            dp = [[0] * (len(slices) // 3 + 1) for _ in xrange(2)]
            for i in xrange(start, end):
                for j in reversed(xrange(1, min((i - start + 1 - 1) // 2 + 1, len(slices) // 3) + 1)):
                    dp[i % 2][j] = max(dp[(i - 1) % 2][j], dp[(i - 2) % 2][j - 1] + slices[i])
            return dp[(end - 1) % 2][len(slices) // 3]
        return max(maxSizeSlicesLinear(slices, 0, len(slices) - 1), maxSizeSlicesLinear(slices, 1, len(slices)))

class Solution2(object):

    def maxSizeSlices(self, slices):
        if False:
            i = 10
            return i + 15
        '\n        :type slices: List[int]\n        :rtype: int\n        '

        def maxSizeSlicesLinear(slices, start, end):
            if False:
                i = 10
                return i + 15
            dp = [[0] * (len(slices) // 3 + 1) for _ in xrange(3)]
            for i in xrange(start, end):
                for j in xrange(1, min((i - start + 1 - 1) // 2 + 1, len(slices) // 3) + 1):
                    dp[i % 3][j] = max(dp[(i - 1) % 3][j], dp[(i - 2) % 3][j - 1] + slices[i])
            return dp[(end - 1) % 3][len(slices) // 3]
        return max(maxSizeSlicesLinear(slices, 0, len(slices) - 1), maxSizeSlicesLinear(slices, 1, len(slices)))