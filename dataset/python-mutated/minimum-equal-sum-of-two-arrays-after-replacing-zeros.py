class Solution(object):

    def minSum(self, nums1, nums2):
        if False:
            return 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        total1 = sum((max(x, 1) for x in nums1))
        total2 = sum((max(x, 1) for x in nums2))
        if total1 < total2:
            return total2 if 0 in nums1 else -1
        if total1 > total2:
            return total1 if 0 in nums2 else -1
        return total1