import operator

class Solution(object):

    def islandPerimeter(self, grid):
        if False:
            i = 10
            return i + 15
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '
        (count, repeat) = (0, 0)
        for i in xrange(len(grid)):
            for j in xrange(len(grid[i])):
                if grid[i][j] == 1:
                    count += 1
                    if i != 0 and grid[i - 1][j] == 1:
                        repeat += 1
                    if j != 0 and grid[i][j - 1] == 1:
                        repeat += 1
        return 4 * count - 2 * repeat

    def islandPerimeter2(self, grid):
        if False:
            while True:
                i = 10
        return sum((sum(map(operator.ne, [0] + row, row + [0])) for row in grid + map(list, zip(*grid))))