class Solution(object):

    def findDisappearedNumbers(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        for i in xrange(len(nums)):
            if nums[abs(nums[i]) - 1] > 0:
                nums[abs(nums[i]) - 1] *= -1
        result = []
        for i in xrange(len(nums)):
            if nums[i] > 0:
                result.append(i + 1)
            else:
                nums[i] *= -1
        return result

    def findDisappearedNumbers2(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        return list(set(range(1, len(nums) + 1)) - set(nums))

    def findDisappearedNumbers3(self, nums):
        if False:
            return 10
        for i in range(len(nums)):
            index = abs(nums[i]) - 1
            nums[index] = -abs(nums[index])
        return [i + 1 for i in range(len(nums)) if nums[i] > 0]