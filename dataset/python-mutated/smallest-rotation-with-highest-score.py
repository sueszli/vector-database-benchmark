class Solution(object):

    def bestRotation(self, A):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        N = len(A)
        change = [1] * N
        for i in xrange(N):
            change[(i - A[i] + 1) % N] -= 1
        for i in xrange(1, N):
            change[i] += change[i - 1]
        return change.index(max(change))