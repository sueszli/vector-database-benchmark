class Solution(object):

    def canThreePartsEqualSum(self, A):
        if False:
            i = 10
            return i + 15
        '\n        :type A: List[int]\n        :rtype: bool\n        '
        total = sum(A)
        if total % 3 != 0:
            return False
        (parts, curr) = (0, 0)
        for x in A:
            curr += x
            if curr == total // 3:
                parts += 1
                curr = 0
        return parts >= 3