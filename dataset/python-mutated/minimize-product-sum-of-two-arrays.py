import itertools
import operator

class Solution(object):

    def minProductSum(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '

        def inner_product(vec1, vec2):
            if False:
                print('Hello World!')
            return sum(itertools.imap(operator.mul, vec1, vec2))
        nums1.sort()
        nums2.sort(reverse=True)
        return inner_product(nums1, nums2)