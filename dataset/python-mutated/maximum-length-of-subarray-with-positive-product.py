class Solution(object):

    def getMaxLen(self, nums):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        (result, neg_cnt, last_zero_pos, first_valid_neg_pos) = (0, 0, -1, -1)
        for i in xrange(len(nums)):
            if nums[i] == 0:
                neg_cnt = 0
                last_zero_pos = i
                first_valid_neg_pos = -1
                continue
            if nums[i] < 0:
                if first_valid_neg_pos == -1:
                    first_valid_neg_pos = i
                neg_cnt += 1
            result = max(result, i - (last_zero_pos if neg_cnt % 2 == 0 else first_valid_neg_pos))
        return result