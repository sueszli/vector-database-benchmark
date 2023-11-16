class Solution(object):

    def smallestEqual(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        return next((i for (i, x) in enumerate(nums) if i % 10 == x), -1)