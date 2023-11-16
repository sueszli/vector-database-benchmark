class Solution(object):

    def diagonalSum(self, mat):
        if False:
            i = 10
            return i + 15
        '\n        :type mat: List[List[int]]\n        :rtype: int\n        '
        return sum((mat[i][i] + mat[~i][i] for i in xrange(len(mat)))) - (mat[len(mat) // 2][len(mat) // 2] if len(mat) % 2 == 1 else 0)