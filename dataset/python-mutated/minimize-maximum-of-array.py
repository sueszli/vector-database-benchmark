class Solution(object):

    def minimizeArrayValue(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                return 10
            return (a + b - 1) // b
        result = curr = 0
        for (i, x) in enumerate(nums):
            curr += x
            result = max(result, ceil_divide(curr, i + 1))
        return result