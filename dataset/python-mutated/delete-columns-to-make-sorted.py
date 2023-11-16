class Solution(object):

    def minDeletionSize(self, A):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[str]\n        :rtype: int\n        '
        result = 0
        for c in xrange(len(A[0])):
            for r in xrange(1, len(A)):
                if A[r - 1][c] > A[r][c]:
                    result += 1
                    break
        return result
import itertools

class Solution2(object):

    def minDeletionSize(self, A):
        if False:
            return 10
        '\n        :type A: List[str]\n        :rtype: int\n        '
        result = 0
        for col in itertools.izip(*A):
            if any((col[i] > col[i + 1] for i in xrange(len(col) - 1))):
                result += 1
        return result