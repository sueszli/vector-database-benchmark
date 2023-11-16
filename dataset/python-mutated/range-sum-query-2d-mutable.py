class NumMatrix(object):

    def __init__(self, matrix):
        if False:
            return 10
        '\n        initialize your data structure here.\n        :type matrix: List[List[int]]\n        '
        if not matrix:
            return
        self.__matrix = matrix
        self.__bit = [[0] * (len(self.__matrix[0]) + 1) for _ in xrange(len(self.__matrix) + 1)]
        for i in xrange(1, len(self.__bit)):
            for j in xrange(1, len(self.__bit[0])):
                self.__bit[i][j] = matrix[i - 1][j - 1] + self.__bit[i - 1][j] + self.__bit[i][j - 1] - self.__bit[i - 1][j - 1]
        for i in reversed(xrange(1, len(self.__bit))):
            for j in reversed(xrange(1, len(self.__bit[0]))):
                (last_i, last_j) = (i - (i & -i), j - (j & -j))
                self.__bit[i][j] = self.__bit[i][j] - self.__bit[i][last_j] - self.__bit[last_i][j] + self.__bit[last_i][last_j]

    def update(self, row, col, val):
        if False:
            i = 10
            return i + 15
        '\n        update the element at matrix[row,col] to val.\n        :type row: int\n        :type col: int\n        :type val: int\n        :rtype: void\n        '
        if val - self.__matrix[row][col]:
            self.__add(row, col, val - self.__matrix[row][col])
            self.__matrix[row][col] = val

    def sumRegion(self, row1, col1, row2, col2):
        if False:
            while True:
                i = 10
        '\n        sum of elements matrix[(row1,col1)..(row2,col2)], inclusive.\n        :type row1: int\n        :type col1: int\n        :type row2: int\n        :type col2: int\n        :rtype: int\n        '
        return self.__sum(row2, col2) - self.__sum(row2, col1 - 1) - self.__sum(row1 - 1, col2) + self.__sum(row1 - 1, col1 - 1)

    def __sum(self, row, col):
        if False:
            return 10
        row += 1
        col += 1
        ret = 0
        i = row
        while i > 0:
            j = col
            while j > 0:
                ret += self.__bit[i][j]
                j -= j & -j
            i -= i & -i
        return ret

    def __add(self, row, col, val):
        if False:
            while True:
                i = 10
        row += 1
        col += 1
        i = row
        while i <= len(self.__matrix):
            j = col
            while j <= len(self.__matrix[0]):
                self.__bit[i][j] += val
                j += j & -j
            i += i & -i