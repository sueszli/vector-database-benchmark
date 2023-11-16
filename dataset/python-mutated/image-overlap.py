class Solution(object):

    def largestOverlap(self, A, B):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[List[int]]\n        :type B: List[List[int]]\n        :rtype: int\n        '
        count = [0] * (2 * len(A) - 1) ** 2
        for (i, row) in enumerate(A):
            for (j, v) in enumerate(row):
                if not v:
                    continue
                for (i2, row2) in enumerate(B):
                    for (j2, v2) in enumerate(row2):
                        if not v2:
                            continue
                        count[(len(A) - 1 + i - i2) * (2 * len(A) - 1) + len(A) - 1 + j - j2] += 1
        return max(count)