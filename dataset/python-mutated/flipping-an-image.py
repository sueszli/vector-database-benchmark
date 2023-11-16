class Solution(object):

    def flipAndInvertImage(self, A):
        if False:
            return 10
        '\n        :type A: List[List[int]]\n        :rtype: List[List[int]]\n        '
        for row in A:
            for i in xrange((len(row) + 1) // 2):
                (row[i], row[~i]) = (row[~i] ^ 1, row[i] ^ 1)
        return A