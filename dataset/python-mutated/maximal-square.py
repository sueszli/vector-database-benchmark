class Solution(object):

    def maximalSquare(self, matrix):
        if False:
            i = 10
            return i + 15
        if not matrix:
            return 0
        (m, n) = (len(matrix), len(matrix[0]))
        size = [[0 for j in xrange(n)] for i in xrange(2)]
        max_size = 0
        for j in xrange(n):
            if matrix[0][j] == '1':
                size[0][j] = 1
            max_size = max(max_size, size[0][j])
        for i in xrange(1, m):
            if matrix[i][0] == '1':
                size[i % 2][0] = 1
            else:
                size[i % 2][0] = 0
            for j in xrange(1, n):
                if matrix[i][j] == '1':
                    size[i % 2][j] = min(size[i % 2][j - 1], size[(i - 1) % 2][j], size[(i - 1) % 2][j - 1]) + 1
                    max_size = max(max_size, size[i % 2][j])
                else:
                    size[i % 2][j] = 0
        return max_size * max_size

class Solution2(object):

    def maximalSquare(self, matrix):
        if False:
            while True:
                i = 10
        if not matrix:
            return 0
        (m, n) = (len(matrix), len(matrix[0]))
        size = [[0 for j in xrange(n)] for i in xrange(m)]
        max_size = 0
        for j in xrange(n):
            if matrix[0][j] == '1':
                size[0][j] = 1
            max_size = max(max_size, size[0][j])
        for i in xrange(1, m):
            if matrix[i][0] == '1':
                size[i][0] = 1
            else:
                size[i][0] = 0
            for j in xrange(1, n):
                if matrix[i][j] == '1':
                    size[i][j] = min(size[i][j - 1], size[i - 1][j], size[i - 1][j - 1]) + 1
                    max_size = max(max_size, size[i][j])
                else:
                    size[i][j] = 0
        return max_size * max_size

class Solution3(object):

    def maximalSquare(self, matrix):
        if False:
            print('Hello World!')
        if not matrix:
            return 0
        (H, W) = (0, 1)
        table = [[[0, 0] for j in xrange(len(matrix[0]))] for i in xrange(len(matrix))]
        for i in reversed(xrange(len(matrix))):
            for j in reversed(xrange(len(matrix[i]))):
                if matrix[i][j] == '1':
                    (h, w) = (1, 1)
                    if i + 1 < len(matrix):
                        h = table[i + 1][j][H] + 1
                    if j + 1 < len(matrix[i]):
                        w = table[i][j + 1][W] + 1
                    table[i][j] = [h, w]
        s = [[0 for j in xrange(len(matrix[0]))] for i in xrange(len(matrix))]
        max_square_area = 0
        for i in reversed(xrange(len(matrix))):
            for j in reversed(xrange(len(matrix[i]))):
                side = min(table[i][j][H], table[i][j][W])
                if matrix[i][j] == '1':
                    if i + 1 < len(matrix) and j + 1 < len(matrix[i + 1]):
                        side = min(s[i + 1][j + 1] + 1, side)
                    s[i][j] = side
                    max_square_area = max(max_square_area, side * side)
        return max_square_area