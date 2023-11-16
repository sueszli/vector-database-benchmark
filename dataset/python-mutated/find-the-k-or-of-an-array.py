class Solution(object):

    def findKOr(self, nums, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        return sum((1 << i for i in xrange(max(nums).bit_length()) if sum((x & 1 << i != 0 for x in nums)) >= k))