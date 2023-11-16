import operator

class Solution(object):

    def xorAllNums(self, nums1, nums2):
        if False:
            while True:
                i = 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '
        return (reduce(operator.xor, nums1) if len(nums2) % 2 else 0) ^ (reduce(operator.xor, nums2) if len(nums1) % 2 else 0)