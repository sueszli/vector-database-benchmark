class Solution(object):

    def largestSubmatrix(self, matrix):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type matrix: List[List[int]]\n        :rtype: int\n        '
        for c in xrange(len(matrix[0])):
            h = 0
            for r in xrange(len(matrix)):
                h = h + 1 if matrix[r][c] == 1 else 0
                matrix[r][c] = h
        result = 0
        for row in matrix:
            row.sort()
            for c in xrange(len(row)):
                result = max(result, (len(row) - c) * row[c])
        return result