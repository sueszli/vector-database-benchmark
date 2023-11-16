class Solution(object):

    def minimumReplacement(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '

        def ceil_divide(a, b):
            if False:
                i = 10
                return i + 15
            return (a + b - 1) // b
        result = 0
        curr = nums[-1]
        for x in reversed(nums):
            cnt = ceil_divide(x, curr)
            result += cnt - 1
            curr = x // cnt
        return result