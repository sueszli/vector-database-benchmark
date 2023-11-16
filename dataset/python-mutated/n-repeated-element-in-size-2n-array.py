class Solution(object):

    def repeatedNTimes(self, A):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        for i in xrange(2, len(A)):
            if A[i - 1] == A[i] or A[i - 2] == A[i]:
                return A[i]
        return A[0]