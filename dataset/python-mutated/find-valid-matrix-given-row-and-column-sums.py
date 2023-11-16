class Solution(object):

    def restoreMatrix(self, rowSum, colSum):
        if False:
            i = 10
            return i + 15
        '\n        :type rowSum: List[int]\n        :type colSum: List[int]\n        :rtype: List[List[int]]\n        '
        matrix = [[0] * len(colSum) for _ in xrange(len(rowSum))]
        i = j = 0
        while i < len(matrix) and j < len(matrix[0]):
            matrix[i][j] = min(rowSum[i], colSum[j])
            rowSum[i] -= matrix[i][j]
            colSum[j] -= matrix[i][j]
            if not rowSum[i]:
                i += 1
            if not colSum[j]:
                j += 1
        return matrix

class Solution2(object):

    def restoreMatrix(self, rowSum, colSum):
        if False:
            i = 10
            return i + 15
        '\n        :type rowSum: List[int]\n        :type colSum: List[int]\n        :rtype: List[List[int]]\n        '
        matrix = [[0] * len(colSum) for _ in xrange(len(rowSum))]
        for i in xrange(len(matrix)):
            for j in xrange(len(matrix[i])):
                matrix[i][j] = min(rowSum[i], colSum[j])
                rowSum[i] -= matrix[i][j]
                colSum[j] -= matrix[i][j]
        return matrix