import collections

class Solution(object):

    def numTriplets(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: int\n        '

        def two_product(nums, i):
            if False:
                print('Hello World!')
            count = 0
            lookup = collections.defaultdict(int)
            for num in nums:
                if i % num:
                    continue
                count += lookup[i // num]
                lookup[num] += 1
            return count
        result = 0
        for num in nums1:
            result += two_product(nums2, num ** 2)
        for num in nums2:
            result += two_product(nums1, num ** 2)
        return result