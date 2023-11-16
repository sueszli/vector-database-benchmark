class Solution(object):

    def nextGreaterElements(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        (result, stk) = ([0] * len(nums), [])
        for i in reversed(xrange(2 * len(nums))):
            while stk and stk[-1] <= nums[i % len(nums)]:
                stk.pop()
            result[i % len(nums)] = stk[-1] if stk else -1
            stk.append(nums[i % len(nums)])
        return result