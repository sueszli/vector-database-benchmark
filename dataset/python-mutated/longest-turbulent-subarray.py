class Solution(object):

    def maxTurbulenceSize(self, A):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        result = 1
        start = 0
        for i in xrange(1, len(A)):
            if i == len(A) - 1 or cmp(A[i - 1], A[i]) * cmp(A[i], A[i + 1]) != -1:
                result = max(result, i - start + 1)
                start = i
        return result