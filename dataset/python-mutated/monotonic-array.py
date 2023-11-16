class Solution(object):

    def isMonotonic(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: bool\n        '
        (inc, dec) = (False, False)
        for i in xrange(len(A) - 1):
            if A[i] < A[i + 1]:
                inc = True
            elif A[i] > A[i + 1]:
                dec = True
        return not inc or not dec