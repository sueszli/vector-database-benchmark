class Solution(object):

    def rowAndMaximumOnes(self, mat):
        if False:
            print('Hello World!')
        '\n        :type mat: List[List[int]]\n        :rtype: List[int]\n        '
        return max(([i, mat[i].count(1)] for i in xrange(len(mat))), key=lambda x: x[1])