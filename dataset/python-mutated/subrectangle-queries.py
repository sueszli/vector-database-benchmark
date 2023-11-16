class SubrectangleQueries(object):

    def __init__(self, rectangle):
        if False:
            print('Hello World!')
        '\n        :type rectangle: List[List[int]]\n        '
        self.__rectangle = rectangle
        self.__updates = []

    def updateSubrectangle(self, row1, col1, row2, col2, newValue):
        if False:
            while True:
                i = 10
        '\n        :type row1: int\n        :type col1: int\n        :type row2: int\n        :type col2: int\n        :type newValue: int\n        :rtype: None\n        '
        self.__updates.append((row1, col1, row2, col2, newValue))

    def getValue(self, row, col):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type row: int\n        :type col: int\n        :rtype: int\n        '
        for (row1, col1, row2, col2, newValue) in reversed(self.__updates):
            if row1 <= row <= row2 and col1 <= col <= col2:
                return newValue
        return self.__rectangle[row][col]

class SubrectangleQueries2(object):

    def __init__(self, rectangle):
        if False:
            return 10
        '\n        :type rectangle: List[List[int]]\n        '
        self.__rectangle = rectangle

    def updateSubrectangle(self, row1, col1, row2, col2, newValue):
        if False:
            print('Hello World!')
        '\n        :type row1: int\n        :type col1: int\n        :type row2: int\n        :type col2: int\n        :type newValue: int\n        :rtype: None\n        '
        for r in xrange(row1, row2 + 1):
            for c in xrange(col1, col2 + 1):
                self.__rectangle[r][c] = newValue

    def getValue(self, row, col):
        if False:
            print('Hello World!')
        '\n        :type row: int\n        :type col: int\n        :rtype: int\n        '
        return self.__rectangle[row][col]