class Solution(object):

    def findRelativeRanks(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: List[str]\n        '
        sorted_nums = sorted(nums)[::-1]
        ranks = ['Gold Medal', 'Silver Medal', 'Bronze Medal'] + map(str, range(4, len(nums) + 1))
        return map(dict(zip(sorted_nums, ranks)).get, nums)