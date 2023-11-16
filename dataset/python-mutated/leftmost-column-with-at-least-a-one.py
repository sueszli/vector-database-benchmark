class BinaryMatrix(object):

    def get(self, row, col):
        if False:
            while True:
                i = 10
        pass

    def dimensions(self):
        if False:
            for i in range(10):
                print('nop')
        pass

class Solution(object):

    def leftMostColumnWithOne(self, binaryMatrix):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type binaryMatrix: BinaryMatrix\n        :rtype: int\n        '
        (m, n) = binaryMatrix.dimensions()
        (r, c) = (0, n - 1)
        while r < m and c >= 0:
            if not binaryMatrix.get(r, c):
                r += 1
            else:
                c -= 1
        return c + 1 if c + 1 != n else -1