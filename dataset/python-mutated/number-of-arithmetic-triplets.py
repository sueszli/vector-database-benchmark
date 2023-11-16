class Solution(object):

    def arithmeticTriplets(self, nums, diff):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type diff: int\n        :rtype: int\n        '
        lookup = set(nums)
        return sum((x - diff in lookup and x - 2 * diff in lookup for x in nums))
import collections

class Solution2(object):

    def arithmeticTriplets(self, nums, diff):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type diff: int\n        :rtype: int\n        '
        result = 0
        cnt1 = collections.Counter()
        cnt2 = collections.Counter()
        for x in nums:
            result += cnt2[x - diff]
            cnt2[x] += cnt1[x - diff]
            cnt1[x] += 1
        return result