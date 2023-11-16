class Solution(object):

    def findColumnWidth(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: List[int]\n        '

        def length(x):
            if False:
                for i in range(10):
                    print('nop')
            l = 1
            if x < 0:
                x = -x
                l += 1
            while x >= 10:
                x //= 10
                l += 1
            return l
        return [max((length(grid[i][j]) for i in xrange(len(grid)))) for j in xrange(len(grid[0]))]

class Solution2(object):

    def findColumnWidth(self, grid):
        if False:
            while True:
                i = 10
        '\n        :type grid: List[List[int]]\n        :rtype: List[int]\n        '
        return [max((len(str(grid[i][j])) for i in xrange(len(grid)))) for j in xrange(len(grid[0]))]
import itertools

class Solution3(object):

    def findColumnWidth(self, grid):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :rtype: List[int]\n        '
        return [max((len(str(x)) for x in col)) for col in itertools.izip(*grid)]