class Solution(object):

    def findPairs(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '
        if k < 0:
            return 0
        (result, lookup) = (set(), set())
        for num in nums:
            if num - k in lookup:
                result.add(num - k)
            if num + k in lookup:
                result.add(num)
            lookup.add(num)
        return len(result)