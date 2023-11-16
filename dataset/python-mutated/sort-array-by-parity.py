class Solution(object):

    def sortArrayByParity(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: List[int]\n        '
        i = 0
        for j in xrange(len(A)):
            if A[j] % 2 == 0:
                (A[i], A[j]) = (A[j], A[i])
                i += 1
        return A