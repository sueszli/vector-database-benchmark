class Solution(object):

    def maxScoreSightseeingPair(self, A):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type A: List[int]\n        :rtype: int\n        '
        (result, curr) = (0, 0)
        for x in A:
            result = max(result, curr + x)
            curr = max(curr, x) - 1
        return result