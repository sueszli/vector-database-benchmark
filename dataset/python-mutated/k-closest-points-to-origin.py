from random import randint

class Solution(object):

    def kClosest(self, points, K):
        if False:
            print('Hello World!')
        '\n        :type points: List[List[int]]\n        :type K: int\n        :rtype: List[List[int]]\n        '

        def dist(point):
            if False:
                return 10
            return point[0] ** 2 + point[1] ** 2

        def kthElement(nums, k, compare):
            if False:
                return 10

            def PartitionAroundPivot(left, right, pivot_idx, nums, compare):
                if False:
                    while True:
                        i = 10
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
                pivot_idx = randint(left, right)
                new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums, compare)
                if new_pivot_idx == k:
                    return
                elif new_pivot_idx > k:
                    right = new_pivot_idx - 1
                else:
                    left = new_pivot_idx + 1
        kthElement(points, K - 1, lambda a, b: dist(a) < dist(b))
        return points[:K]
import heapq

class Solution2(object):

    def kClosest(self, points, K):
        if False:
            print('Hello World!')
        '\n        :type points: List[List[int]]\n        :type K: int\n        :rtype: List[List[int]]\n        '

        def dist(point):
            if False:
                while True:
                    i = 10
            return point[0] ** 2 + point[1] ** 2
        max_heap = []
        for point in points:
            heapq.heappush(max_heap, (-dist(point), point))
            if len(max_heap) > K:
                heapq.heappop(max_heap)
        return [heapq.heappop(max_heap)[1] for _ in xrange(len(max_heap))]