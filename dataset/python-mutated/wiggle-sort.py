class Solution(object):

    def wiggleSort(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: void Do not return anything, modify nums in-place instead.\n        '
        for i in xrange(1, len(nums)):
            if i % 2 and nums[i - 1] > nums[i] or (not i % 2 and nums[i - 1] < nums[i]):
                (nums[i - 1], nums[i]) = (nums[i], nums[i - 1])

class Solution2(object):

    def wiggleSort(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: void Do not return anything, modify nums in-place instead.\n        '
        nums.sort()
        med = (len(nums) - 1) // 2
        (nums[::2], nums[1::2]) = (nums[med::-1], nums[:med:-1])