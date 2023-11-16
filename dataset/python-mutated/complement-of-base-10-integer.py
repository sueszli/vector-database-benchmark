class Solution(object):

    def bitwiseComplement(self, N):
        if False:
            i = 10
            return i + 15
        '\n        :type N: int\n        :rtype: int\n        '
        mask = 1
        while N > mask:
            mask = mask * 2 + 1
        return mask - N