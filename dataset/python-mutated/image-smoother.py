class Solution(object):

    def imageSmoother(self, M):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type M: List[List[int]]\n        :rtype: List[List[int]]\n        '

        def getGray(M, i, j):
            if False:
                i = 10
                return i + 15
            (total, count) = (0, 0.0)
            for r in xrange(-1, 2):
                for c in xrange(-1, 2):
                    (ii, jj) = (i + r, j + c)
                    if 0 <= ii < len(M) and 0 <= jj < len(M[0]):
                        total += M[ii][jj]
                        count += 1.0
            return int(total / count)
        result = [[0 for _ in xrange(len(M[0]))] for _ in xrange(len(M))]
        for i in xrange(len(M)):
            for j in xrange(len(M[0])):
                result[i][j] = getGray(M, i, j)
        return result