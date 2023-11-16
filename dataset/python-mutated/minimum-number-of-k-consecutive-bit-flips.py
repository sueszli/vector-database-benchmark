class Solution(object):

    def minKBitFlips(self, A, K):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[int]\n        :type K: int\n        :rtype: int\n        '
        (result, curr) = (0, 0)
        for i in xrange(len(A)):
            if i >= K:
                curr -= A[i - K] // 2
            if curr & 1 ^ A[i] == 0:
                if i + K > len(A):
                    return -1
                A[i] += 2
                (curr, result) = (curr + 1, result + 1)
        return result