import collections

class Solution(object):

    def intersect(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1)
        lookup = collections.defaultdict(int)
        for i in nums1:
            lookup[i] += 1
        res = []
        for i in nums2:
            if lookup[i] > 0:
                res += (i,)
                lookup[i] -= 1
        return res

    def intersect2(self, nums1, nums2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        c = collections.Counter(nums1) & collections.Counter(nums2)
        intersect = []
        for i in c:
            intersect.extend([i] * c[i])
        return intersect

class Solution(object):

    def intersect(self, nums1, nums2):
        if False:
            i = 10
            return i + 15
        '\n        :type nums1: List[int]\n        :type nums2: List[int]\n        :rtype: List[int]\n        '
        if len(nums1) > len(nums2):
            return self.intersect(nums2, nums1)

        def binary_search(compare, nums, left, right, target):
            if False:
                i = 10
                return i + 15
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
                left += 1
        return res

class Solution(object):

    def intersect(self, nums1, nums2):
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
                res += (nums1[it1],)
                it1 += 1
                it2 += 1
        return res

class Solution(object):

    def intersect(self, nums1, nums2):
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
                res += (nums1[it1],)
                it1 += 1
                it2 += 1
        return res