import bisect

class Solution(object):

    def matrixMedian(self, grid):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :rtype: int\n        '

        def check(x):
            if False:
                while True:
                    i = 10
            return sum((bisect_right(row, x) for row in grid)) > len(grid) * len(grid[0]) // 2
        (left, right) = (min((row[0] for row in grid)), max((row[-1] for row in grid)))
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left