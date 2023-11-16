class Solution(object):

    def solveNQueens(self, n):
        if False:
            i = 10
            return i + 15
        '\n        :type n: int\n        :rtype: List[List[str]]\n        '

        def dfs(row):
            if False:
                i = 10
                return i + 15
            if row == n:
                result.append(map(lambda x: '.' * x + 'Q' + '.' * (n - x - 1), curr))
                return
            for i in xrange(n):
                if cols[i] or main_diag[row + i] or anti_diag[row - i + (n - 1)]:
                    continue
                cols[i] = main_diag[row + i] = anti_diag[row - i + (n - 1)] = True
                curr.append(i)
                dfs(row + 1)
                curr.pop()
                cols[i] = main_diag[row + i] = anti_diag[row - i + (n - 1)] = False
        (result, curr) = ([], [])
        (cols, main_diag, anti_diag) = ([False] * n, [False] * (2 * n - 1), [False] * (2 * n - 1))
        dfs(0)
        return result

class Solution2(object):

    def solveNQueens(self, n):
        if False:
            while True:
                i = 10
        '\n        :type n: int\n        :rtype: List[List[str]]\n        '

        def dfs(col_per_row, xy_diff, xy_sum):
            if False:
                return 10
            cur_row = len(col_per_row)
            if cur_row == n:
                ress.append(col_per_row)
            for col in range(n):
                if col not in col_per_row and cur_row - col not in xy_diff and (cur_row + col not in xy_sum):
                    dfs(col_per_row + [col], xy_diff + [cur_row - col], xy_sum + [cur_row + col])
        ress = []
        dfs([], [], [])
        return [['.' * i + 'Q' + '.' * (n - i - 1) for i in res] for res in ress]