class Solution(object):

    def prefixesDivBy5(self, A):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :rtype: List[bool]\n        '
        for i in xrange(1, len(A)):
            A[i] += A[i - 1] * 2 % 5
        return [x % 5 == 0 for x in A]