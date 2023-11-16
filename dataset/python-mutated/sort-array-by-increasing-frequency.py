import collections

class Solution(object):

    def frequencySort(self, nums):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :rtype: List[int]\n        '
        count = collections.Counter(nums)
        return sorted(nums, key=lambda x: (count[x], -x))