import bisect

class Solution(object):

    def minAbsoluteSumDiff(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        MOD = 10 ** 9 + 7
        sorted_nums1 = sorted(nums1)
        result = max_change = 0
        for i in xrange(len(nums2)):
            diff = abs(nums1[i] - nums2[i])
            result = (result + diff) % MOD
            if diff < max_change:
                continue
            j = bisect.bisect_left(sorted_nums1, nums2[i])
            if j != len(sorted_nums1):
                max_change = max(max_change, diff - abs(sorted_nums1[j] - nums2[i]))
            if j != 0:
                max_change = max(max_change, diff - abs(sorted_nums1[j - 1] - nums2[i]))
        return (result - max_change) % MOD