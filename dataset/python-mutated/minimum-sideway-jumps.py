class Solution(object):

    def minSideJumps(self, obstacles):
        if False:
            while True:
                i = 10
        '\n        :type obstacles: List[int]\n        :rtype: int\n        '
        (result, lanes) = (0, set([2]))
        for i in xrange(len(obstacles) - 1):
            lanes.discard(obstacles[i + 1])
            if lanes:
                continue
            result += 1
            lanes = set((j for j in xrange(1, 4) if j not in [obstacles[i], obstacles[i + 1]]))
        return result

class Solution2(object):

    def minSideJumps(self, obstacles):
        if False:
            print('Hello World!')
        '\n        :type obstacles: List[int]\n        :rtype: int\n        '
        dp = [1, 0, 1]
        for i in obstacles:
            if i:
                dp[i - 1] = float('inf')
            for j in xrange(3):
                if j + 1 != i:
                    dp[j] = min(dp[0] + (j != 0), dp[1] + (j != 1), dp[2] + (j != 2))
        return min(dp)