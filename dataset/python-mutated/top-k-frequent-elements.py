import collections

class Solution(object):

    def topKFrequent(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        counts = collections.Counter(nums)
        buckets = [[] for _ in xrange(len(nums) + 1)]
        for (i, count) in counts.iteritems():
            buckets[count].append(i)
        result = []
        for i in reversed(xrange(len(buckets))):
            for j in xrange(len(buckets[i])):
                result.append(buckets[i][j])
                if len(result) == k:
                    return result
        return result
from random import randint

class Solution2(object):

    def topKFrequent(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        counts = collections.Counter(nums)
        p = []
        for (key, val) in counts.iteritems():
            p.append((-val, key))
        self.kthElement(p, k - 1)
        result = []
        for i in xrange(k):
            result.append(p[i][1])
        return result

    def kthElement(self, nums, k):
        if False:
            for i in range(10):
                print('nop')

        def PartitionAroundPivot(left, right, pivot_idx, nums):
            if False:
                i = 10
                return i + 15
            pivot_value = nums[pivot_idx]
            new_pivot_idx = left
            (nums[pivot_idx], nums[right]) = (nums[right], nums[pivot_idx])
            for i in xrange(left, right):
                if nums[i] < pivot_value:
                    (nums[i], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[i])
                    new_pivot_idx += 1
            (nums[right], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[right])
            return new_pivot_idx
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            pivot_idx = randint(left, right)
            new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums)
            if new_pivot_idx == k:
                return
            elif new_pivot_idx > k:
                right = new_pivot_idx - 1
            else:
                left = new_pivot_idx + 1

class Solution3(object):

    def topKFrequent(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        return [key for (key, _) in collections.Counter(nums).most_common(k)]