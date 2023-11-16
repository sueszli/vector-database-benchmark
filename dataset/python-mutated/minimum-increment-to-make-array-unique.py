class Solution(object):

    def minIncrementForUnique(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: int\n        '
        A.sort()
        A.append(float('inf'))
        (result, duplicate) = (0, 0)
        for i in xrange(1, len(A)):
            if A[i - 1] == A[i]:
                duplicate += 1
                result -= A[i]
            else:
                move = min(duplicate, A[i] - A[i - 1] - 1)
                duplicate -= move
                result += move * A[i - 1] + move * (move + 1) // 2
        return result