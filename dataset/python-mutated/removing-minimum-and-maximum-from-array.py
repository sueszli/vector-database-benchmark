class Solution(object):

    def minimumDeletions(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (i, j) = (nums.index(min(nums)), nums.index(max(nums)))
        if i > j:
            (i, j) = (j, i)
        return min(i + 1 + (len(nums) - j), j + 1, len(nums) - i)