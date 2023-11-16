import collections

class Solution(object):

    def countExcellentPairs(self, nums, k):
        if False:
            return 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def popcount(x):
            if False:
                for i in range(10):
                    print('nop')
            return bin(x)[2:].count('1')
        cnt = collections.Counter((popcount(x) for x in set(nums)))
        return sum((cnt[i] * cnt[j] for i in cnt.iterkeys() for j in cnt.iterkeys() if i + j >= k))

class Solution2(object):

    def countExcellentPairs(self, nums, k):
        if False:
            i = 10
            return i + 15
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: int\n        '

        def popcount(x):
            if False:
                i = 10
                return i + 15
            return bin(x)[2:].count('1')
        sorted_cnts = sorted((popcount(x) for x in set(nums)))
        result = 0
        (left, right) = (0, len(sorted_cnts) - 1)
        while left <= right:
            if sorted_cnts[left] + sorted_cnts[right] < k:
                left += 1
            else:
                result += 1 + 2 * (right - 1 - left + 1)
                right -= 1
        return result