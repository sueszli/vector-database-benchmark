class Solution(object):

    def maxHeight(self, cuboids):
        if False:
            print('Hello World!')
        '\n        :type cuboids: List[List[int]]\n        :rtype: int\n        '
        for cuboid in cuboids:
            cuboid.sort()
        cuboids.append([0, 0, 0])
        cuboids.sort()
        dp = [0] * len(cuboids)
        for i in xrange(1, len(cuboids)):
            for j in xrange(i):
                if all((cuboids[j][k] <= cuboids[i][k] for k in xrange(3))):
                    dp[i] = max(dp[i], dp[j] + cuboids[i][2])
        return max(dp)