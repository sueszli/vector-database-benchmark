class Solution(object):

    def kWeakestRows(self, mat, k):
        if False:
            return 10
        '\n        :type mat: List[List[int]]\n        :type k: int\n        :rtype: List[int]\n        '
        (result, lookup) = ([], set())
        for j in xrange(len(mat[0])):
            for i in xrange(len(mat)):
                if mat[i][j] or i in lookup:
                    continue
                lookup.add(i)
                result.append(i)
                if len(result) == k:
                    return result
        for i in xrange(len(mat)):
            if i in lookup:
                continue
            lookup.add(i)
            result.append(i)
            if len(result) == k:
                break
        return result
import collections

class Solution2(object):

    def kWeakestRows(self, mat, k):
        if False:
            while True:
                i = 10
        '\n        :type mat: List[List[int]]\n        :type k: int\n        :rtype: List[int]\n        '
        lookup = collections.OrderedDict()
        for j in xrange(len(mat[0])):
            for i in xrange(len(mat)):
                if mat[i][j] or i in lookup:
                    continue
                lookup[i] = True
                if len(lookup) == k:
                    return lookup.keys()
        for i in xrange(len(mat)):
            if i in lookup:
                continue
            lookup[i] = True
            if len(lookup) == k:
                break
        return lookup.keys()
import random

class Solution3(object):

    def kWeakestRows(self, mat, k):
        if False:
            print('Hello World!')
        '\n        :type mat: List[List[int]]\n        :type k: int\n        :rtype: List[int]\n        '

        def nth_element(nums, n, compare=lambda a, b: a < b):
            if False:
                i = 10
                return i + 15

            def partition_around_pivot(left, right, pivot_idx, nums, compare):
                if False:
                    i = 10
                    return i + 15
                new_pivot_idx = left
                (nums[pivot_idx], nums[right]) = (nums[right], nums[pivot_idx])
                for i in xrange(left, right):
                    if compare(nums[i], nums[right]):
                        (nums[i], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[i])
                        new_pivot_idx += 1
                (nums[right], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[right])
                return new_pivot_idx
            (left, right) = (0, len(nums) - 1)
            while left <= right:
                pivot_idx = random.randint(left, right)
                new_pivot_idx = partition_around_pivot(left, right, pivot_idx, nums, compare)
                if new_pivot_idx == n:
                    return
                elif new_pivot_idx > n:
                    right = new_pivot_idx - 1
                else:
                    left = new_pivot_idx + 1
        nums = [(sum(mat[i]), i) for i in xrange(len(mat))]
        nth_element(nums, k)
        return map(lambda x: x[1], sorted(nums[:k]))