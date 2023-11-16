class Solution(object):

    def deleteAndEarn(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        vals = [0] * 10001
        for num in nums:
            vals[num] += num
        (val_i, val_i_1) = (vals[0], 0)
        for i in xrange(1, len(vals)):
            (val_i_1, val_i_2) = (val_i, val_i_1)
            val_i = max(vals[i] + val_i_2, val_i_1)
        return val_i