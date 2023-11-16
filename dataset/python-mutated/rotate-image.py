class Solution(object):

    def rotate(self, matrix):
        if False:
            while True:
                i = 10
        n = len(matrix)
        for i in xrange(n):
            for j in xrange(n - i):
                (matrix[i][j], matrix[n - 1 - j][n - 1 - i]) = (matrix[n - 1 - j][n - 1 - i], matrix[i][j])
        for i in xrange(n / 2):
            for j in xrange(n):
                (matrix[i][j], matrix[n - 1 - i][j]) = (matrix[n - 1 - i][j], matrix[i][j])
        return matrix

class Solution2(object):

    def rotate(self, matrix):
        if False:
            print('Hello World!')
        return [list(reversed(x)) for x in zip(*matrix)]