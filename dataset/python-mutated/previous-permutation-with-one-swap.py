class Solution(object):

    def prevPermOpt1(self, A):
        if False:
            return 10
        '\n        :type A: List[int]\n        :rtype: List[int]\n        '
        for left in reversed(xrange(len(A) - 1)):
            if A[left] > A[left + 1]:
                break
        else:
            return A
        right = len(A) - 1
        while A[left] <= A[right]:
            right -= 1
        while A[right - 1] == A[right]:
            right -= 1
        (A[left], A[right]) = (A[right], A[left])
        return A