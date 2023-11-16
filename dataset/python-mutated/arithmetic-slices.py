class Solution(object):

    def numberOfArithmeticSlices(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: int\n        '
        (res, i) = (0, 0)
        while i + 2 < len(A):
            start = i
            while i + 2 < len(A) and A[i + 2] + A[i] == 2 * A[i + 1]:
                res += i - start + 1
                i += 1
            i += 1
        return res