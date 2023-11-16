class Solution(object):

    def minNumber(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        common = set(nums1) & set(nums2)
        if common:
            return min(common)
        (mn1, mn2) = (min(nums1), min(nums2))
        if mn1 > mn2:
            (mn1, mn2) = (mn2, mn1)
        return 10 * mn1 + mn2