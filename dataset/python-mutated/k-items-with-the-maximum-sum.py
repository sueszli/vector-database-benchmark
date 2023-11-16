class Solution(object):

    def kItemsWithMaximumSum(self, numOnes, numZeros, numNegOnes, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type numOnes: int\n        :type numZeros: int\n        :type numNegOnes: int\n        :type k: int\n        :rtype: int\n        '
        return min(numOnes, k) - max(k - numOnes - numZeros, 0)