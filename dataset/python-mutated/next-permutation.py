class Solution(object):

    def nextPermutation(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: None Do not return anything, modify nums in-place instead.\n        '
        (k, l) = (-1, 0)
        for i in reversed(xrange(len(nums) - 1)):
            if nums[i] < nums[i + 1]:
                k = i
                break
        else:
            nums.reverse()
            return
        for i in reversed(xrange(k + 1, len(nums))):
            if nums[i] > nums[k]:
                l = i
                break
        (nums[k], nums[l]) = (nums[l], nums[k])
        nums[k + 1:] = nums[:k:-1]

class Solution2(object):

    def nextPermutation(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: None Do not return anything, modify nums in-place instead.\n        '
        (k, l) = (-1, 0)
        for i in xrange(len(nums) - 1):
            if nums[i] < nums[i + 1]:
                k = i
        if k == -1:
            nums.reverse()
            return
        for i in xrange(k + 1, len(nums)):
            if nums[i] > nums[k]:
                l = i
        (nums[k], nums[l]) = (nums[l], nums[k])
        nums[k + 1:] = nums[:k:-1]