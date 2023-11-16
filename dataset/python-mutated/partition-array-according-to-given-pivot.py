class Solution(object):

    def pivotArray(self, nums, pivot):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type pivot: int\n        :rtype: List[int]\n        '
        result = [pivot] * len(nums)
        (left, right) = (0, len(nums) - sum((x > pivot for x in nums)))
        for x in nums:
            if x < pivot:
                result[left] = x
                left += 1
            elif x > pivot:
                result[right] = x
                right += 1
        return result