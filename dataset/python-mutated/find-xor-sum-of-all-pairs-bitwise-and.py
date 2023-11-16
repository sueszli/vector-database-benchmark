import operator

class Solution(object):

    def getXORSum(self, arr1, arr2):
        if False:
            i = 10
            return i + 15
        '\n        :type arr1: List[int]\n        :type arr2: List[int]\n        :rtype: int\n        '
        return reduce(operator.xor, arr1) & reduce(operator.xor, arr2)