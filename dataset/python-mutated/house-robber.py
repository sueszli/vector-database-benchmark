class Solution(object):

    def rob(self, nums):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (last, now) = (0, 0)
        for i in nums:
            (last, now) = (now, max(last + i, now))
        return now