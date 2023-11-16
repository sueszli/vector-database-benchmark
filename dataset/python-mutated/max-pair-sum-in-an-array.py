class Solution(object):

    def maxSum(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def max_digit(x):
            if False:
                i = 10
                return i + 15
            result = 0
            while x:
                (x, r) = divmod(x, 10)
                result = max(result, r)
            return result
        result = -1
        lookup = {}
        for x in nums:
            mx = max_digit(x)
            if mx not in lookup:
                lookup[mx] = x
                continue
            result = max(result, lookup[mx] + x)
            lookup[mx] = max(lookup[mx], x)
        return result