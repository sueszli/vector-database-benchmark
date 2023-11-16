class Solution(object):

    def maxRotateFunction(self, A):
        if False:
            print('Hello World!')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        s = sum(A)
        fi = 0
        for i in xrange(len(A)):
            fi += i * A[i]
        result = fi
        for i in xrange(1, len(A) + 1):
            fi += s - len(A) * A[-i]
            result = max(result, fi)
        return result