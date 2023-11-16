class Solution(object):

    def intersection(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        if len(nums1) > len(nums2):
            return self.intersection(nums2, nums1)
        lookup = set()
        for i in nums1:
            lookup.add(i)
        res = []
        for i in nums2:
            if i in lookup:
                res += (i,)
                lookup.discard(i)
        return res

    def intersection2(self, nums1, nums2):
        if False:
            return 10
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        return list(set(nums1) & set(nums2))

class Solution2(object):

    def intersection(self, nums1, nums2):
        if False:
            print('Hello World!')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        if len(nums1) > len(nums2):
            return self.intersection(nums2, nums1)

        def binary_search(compare, nums, left, right, target):
            if False:
                print('Hello World!')
            while left < right:
                mid = left + (right - left) / 2
                if compare(nums[mid], target):
                    right = mid
                else:
                    left = mid + 1
            return left
        (nums1.sort(), nums2.sort())
        res = []
        left = 0
        for i in nums1:
            left = binary_search(lambda x, y: x >= y, nums2, left, len(nums2), i)
            if left != len(nums2) and nums2[left] == i:
                res += (i,)
                left = binary_search(lambda x, y: x > y, nums2, left, len(nums2), i)
        return res

class Solution3(object):

    def intersection(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        (nums1.sort(), nums2.sort())
        res = []
        (it1, it2) = (0, 0)
        while it1 < len(nums1) and it2 < len(nums2):
            if nums1[it1] < nums2[it2]:
                it1 += 1
            elif nums1[it1] > nums2[it2]:
                it2 += 1
            else:
                if not res or res[-1] != nums1[it1]:
                    res += (nums1[it1],)
                it1 += 1
                it2 += 1
        return res