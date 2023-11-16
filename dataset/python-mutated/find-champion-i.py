class Solution(object):

    def findChampion(self, grid):
        if False:
            return 10
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        return next((u for u in xrange(len(grid)) if sum(grid[u]) == len(grid) - 1))