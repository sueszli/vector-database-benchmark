class Solution(object):

    def longestOnes(self, A, K):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :type K: int\n        :rtype: int\n        '
        (result, i) = (0, 0)
        for j in xrange(len(A)):
            K -= int(A[j] == 0)
            while K < 0:
                K += int(A[i] == 0)
                i += 1
            result = max(result, j - i + 1)
        return result