import collections

class Solution(object):

    def distinctNumbers(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[int]\n        '
        result = []
        count = collections.Counter()
        for (i, num) in enumerate(nums):
            count[num] += 1
            if i >= k:
                count[nums[i - k]] -= 1
                if not count[nums[i - k]]:
                    del count[nums[i - k]]
            if i + 1 >= k:
                result.append(len(count))
        return result