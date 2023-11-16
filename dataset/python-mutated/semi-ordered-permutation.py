class Solution(object):

    def semiOrderedPermutation(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (i, j) = (nums.index(1), nums.index(len(nums)))
        return i + (len(nums) - 1 - j) - int(i > j)