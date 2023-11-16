import collections

class Solution(object):

    def mostFrequent(self, nums, key):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type key: int\n        :rtype: int\n        '
        return collections.Counter((nums[i + 1] for i in xrange(len(nums) - 1) if nums[i] == key)).most_common(1)[0][0]