class Solution(object):

    def maxSubarrayLength(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        stk = []
        for i in reversed(xrange(len(nums))):
            if not stk or nums[stk[-1]] > nums[i]:
                stk.append(i)
        result = 0
        for left in xrange(len(nums)):
            while stk and nums[stk[-1]] < nums[left]:
                result = max(result, stk.pop() - left + 1)
        return result

class Solution2(object):

    def maxSubarrayLength(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        idxs = range(len(nums))
        idxs.sort(key=lambda x: nums[x], reverse=True)
        result = 0
        for left in xrange(len(nums)):
            while idxs and nums[idxs[-1]] < nums[left]:
                result = max(result, idxs.pop() - left + 1)
        return result