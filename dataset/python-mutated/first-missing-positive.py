class Solution(object):

    def firstMissingPositive(self, A):
        if False:
            return 10
        i = 0
        while i < len(A):
            if A[i] > 0 and A[i] - 1 < len(A) and (A[i] != A[A[i] - 1]):
                (A[A[i] - 1], A[i]) = (A[i], A[A[i] - 1])
            else:
                i += 1
        for (i, integer) in enumerate(A):
            if integer != i + 1:
                return i + 1
        return len(A) + 1