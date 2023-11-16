class Solution(object):

    def maximumGap(self, nums):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        if len(nums) < 2:
            return 0
        (max_val, min_val) = (max(nums), min(nums))
        gap = max(1, (max_val - min_val) / (len(nums) - 1))
        bucket_size = (max_val - min_val) / gap + 1
        bucket = [{'min': float('inf'), 'max': float('-inf')} for _ in xrange(bucket_size)]
        for n in nums:
            if n in (max_val, min_val):
                continue
            i = (n - min_val) / gap
            bucket[i]['min'] = min(bucket[i]['min'], n)
            bucket[i]['max'] = max(bucket[i]['max'], n)
        (max_gap, pre_bucket_max) = (0, min_val)
        for i in xrange(bucket_size):
            if bucket[i]['min'] == float('inf') and bucket[i]['max'] == float('-inf'):
                continue
            max_gap = max(max_gap, bucket[i]['min'] - pre_bucket_max)
            pre_bucket_max = bucket[i]['max']
        max_gap = max(max_gap, max_val - pre_bucket_max)
        return max_gap

class Solution2(object):

    def maximumGap(self, nums):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :rtype: int\n        '
        if len(nums) < 2:
            return 0
        nums.sort()
        pre = nums[0]
        max_gap = float('-inf')
        for i in nums:
            max_gap = max(max_gap, i - pre)
            pre = i
        return max_gap