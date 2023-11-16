class NumMatrix(object):

    def __init__(self, matrix):
        if False:
            while True:
                i = 10
        '\n        initialize your data structure here.\n        :type matrix: List[List[int]]\n        '
        if not matrix:
            return
        (m, n) = (len(matrix), len(matrix[0]))
        self.__sums = [[0 for _ in xrange(n + 1)] for _ in xrange(m + 1)]
        for i in xrange(1, m + 1):
            for j in xrange(1, n + 1):
                self.__sums[i][j] = self.__sums[i][j - 1] + self.__sums[i - 1][j] - self.__sums[i - 1][j - 1] + matrix[i - 1][j - 1]

    def sumRegion(self, row1, col1, row2, col2):
        if False:
            print('Hello World!')
        '\n        sum of elements matrix[(row1,col1)..(row2,col2)], inclusive.\n        :type row1: int\n        :type col1: int\n        :type row2: int\n        :type col2: int\n        :rtype: int\n        '
        return self.__sums[row2 + 1][col2 + 1] - self.__sums[row2 + 1][col1] - self.__sums[row1][col2 + 1] + self.__sums[row1][col1]