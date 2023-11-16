class Solution(object):

    def moveZeroes(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: void Do not return anything, modify nums in-place instead.\n        '
        pos = 0
        for i in xrange(len(nums)):
            if nums[i]:
                (nums[i], nums[pos]) = (nums[pos], nums[i])
                pos += 1

    def moveZeroes2(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: void Do not return anything, modify nums in-place instead.\n        '
        nums.sort(cmp=lambda a, b: 0 if b else -1)

class Solution2(object):

    def moveZeroes(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: void Do not return anything, modify nums in-place instead.\n        '
        pos = 0
        for i in xrange(len(nums)):
            if nums[i]:
                nums[pos] = nums[i]
                pos += 1
        for i in xrange(pos, len(nums)):
            nums[i] = 0