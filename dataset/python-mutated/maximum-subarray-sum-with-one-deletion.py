class Solution(object):

    def maximumSum(self, arr):
        if False:
            while True:
                i = 10
        '\n        :type arr: List[int]\n        :rtype: int\n        '
        (result, prev, curr) = (float('-inf'), float('-inf'), float('-inf'))
        for x in arr:
            curr = max(prev, curr + x, x)
            result = max(result, curr)
            prev = max(prev + x, x)
        return result