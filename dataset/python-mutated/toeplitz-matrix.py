class Solution(object):

    def isToeplitzMatrix(self, matrix):
        if False:
            while True:
                i = 10
        '\n        :type matrix: List[List[int]]\n        :rtype: bool\n        '
        return all((i == 0 or j == 0 or matrix[i - 1][j - 1] == val for (i, row) in enumerate(matrix) for (j, val) in enumerate(row)))

class Solution2(object):

    def isToeplitzMatrix(self, matrix):
        if False:
            print('Hello World!')
        '\n        :type matrix: List[List[int]]\n        :rtype: bool\n        '
        for (row_index, row) in enumerate(matrix):
            for (digit_index, digit) in enumerate(row):
                if not row_index or not digit_index:
                    continue
                if matrix[row_index - 1][digit_index - 1] != digit:
                    return False
        return True