import collections
import random

class Leaderboard(object):

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.__lookup = collections.Counter()

    def addScore(self, playerId, score):
        if False:
            return 10
        '\n        :type playerId: int\n        :type score: int\n        :rtype: None\n        '
        self.__lookup[playerId] += score

    def top(self, K):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type K: int\n        :rtype: int\n        '

        def kthElement(nums, k, compare):
            if False:
                i = 10
                return i + 15

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
                pivot_idx = random.randint(left, right)
                new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums, compare)
                if new_pivot_idx == k:
                    return
                elif new_pivot_idx > k:
                    right = new_pivot_idx - 1
                else:
                    left = new_pivot_idx + 1
        scores = self.__lookup.values()
        kthElement(scores, K, lambda a, b: a > b)
        return sum(scores[:K])

    def reset(self, playerId):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type playerId: int\n        :rtype: None\n        '
        self.__lookup[playerId] = 0