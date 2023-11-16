class Solution(object):

    def isIdealPermutation(self, A):
        if False:
            return 10
        '\n        :type A: List[int]\n        :rtype: bool\n        '
        return all((abs(v - i) <= 1 for (i, v) in enumerate(A)))