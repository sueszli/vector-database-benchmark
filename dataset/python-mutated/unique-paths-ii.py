class Solution(object):

    def uniquePathsWithObstacles(self, obstacleGrid):
        if False:
            print('Hello World!')
        '\n        :type obstacleGrid: List[List[int]]\n        :rtype: int\n        '
        (m, n) = (len(obstacleGrid), len(obstacleGrid[0]))
        ways = [0] * n
        ways[0] = 1
        for i in xrange(m):
            if obstacleGrid[i][0] == 1:
                ways[0] = 0
            for j in xrange(n):
                if obstacleGrid[i][j] == 1:
                    ways[j] = 0
                elif j > 0:
                    ways[j] += ways[j - 1]
        return ways[-1]