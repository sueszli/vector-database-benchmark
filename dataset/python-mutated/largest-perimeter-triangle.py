class Solution(object):

    def largestPerimeter(self, A):
        if False:
            while True:
                i = 10
        '\n        :type A: List[int]\n        :rtype: int\n        '
        A.sort()
        for i in reversed(xrange(len(A) - 2)):
            if A[i] + A[i + 1] > A[i + 2]:
                return A[i] + A[i + 1] + A[i + 2]
        return 0