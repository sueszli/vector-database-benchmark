class Solution(object):

    def sortArrayByParityII(self, A):
        if False:
            return 10
        '\n        :type A: List[int]\n        :rtype: List[int]\n        '
        j = 1
        for i in xrange(0, len(A), 2):
            if A[i] % 2:
                while A[j] % 2:
                    j += 2
                (A[i], A[j]) = (A[j], A[i])
        return A