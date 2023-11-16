class Solution(object):

    def shiftGrid(self, grid, k):
        if False:
            print('Hello World!')
        '\n        :type grid: List[List[int]]\n        :type k: int\n        :rtype: List[List[int]]\n        '

        def rotate(grids, k):
            if False:
                return 10

            def reverse(grid, start, end):
                if False:
                    print('Hello World!')
                while start < end:
                    (start_r, start_c) = divmod(start, len(grid[0]))
                    (end_r, end_c) = divmod(end - 1, len(grid[0]))
                    (grid[start_r][start_c], grid[end_r][end_c]) = (grid[end_r][end_c], grid[start_r][start_c])
                    start += 1
                    end -= 1
            k %= len(grid) * len(grid[0])
            reverse(grid, 0, len(grid) * len(grid[0]))
            reverse(grid, 0, k)
            reverse(grid, k, len(grid) * len(grid[0]))
        rotate(grid, k)
        return grid