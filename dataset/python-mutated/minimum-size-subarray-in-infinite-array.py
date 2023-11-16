class Solution(object):

    def minSizeSubarray(self, nums, target):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        INF = float('inf')
        (q, target) = divmod(target, sum(nums))
        if not target:
            return q * len(nums)
        result = INF
        curr = left = 0
        for right in xrange(len(nums) - 1 + (len(nums) - 1)):
            curr += nums[right % len(nums)]
            while curr > target:
                curr -= nums[left % len(nums)]
                left += 1
            if curr == target:
                result = min(result, right - left + 1)
        return result + q * len(nums) if result != INF else -1

class Solution2(object):

    def minSizeSubarray(self, nums, target):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type target: int\n        :rtype: int\n        '
        INF = float('inf')
        (q, target) = divmod(target, sum(nums))
        if not target:
            return q * len(nums)
        result = INF
        lookup = {0: -1}
        prefix = 0
        for right in xrange(len(nums) - 1 + (len(nums) - 1)):
            prefix += nums[right % len(nums)]
            if prefix - target in lookup:
                result = min(result, right - lookup[prefix - target])
            lookup[prefix] = right
        return result + q * len(nums) if result != INF else -1