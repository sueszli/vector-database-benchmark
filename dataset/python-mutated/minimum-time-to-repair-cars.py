import collections

class Solution(object):

    def repairCars(self, ranks, cars):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type ranks: List[int]\n        :type cars: int\n        :rtype: int\n        '

        def check(x):
            if False:
                while True:
                    i = 10
            return sum((int((x // k) ** 0.5) * v for (k, v) in cnt.iteritems())) >= cars
        cnt = collections.Counter(ranks)
        (left, right) = (1, min(cnt.iterkeys()) * cars ** 2)
        while left <= right:
            mid = left + (right - left) // 2
            if check(mid):
                right = mid - 1
            else:
                left = mid + 1
        return left
import collections
import heapq

class Solution2(object):

    def repairCars(self, ranks, cars):
        if False:
            i = 10
            return i + 15
        '\n        :type ranks: List[int]\n        :type cars: int\n        :rtype: int\n        '
        cnt = collections.Counter(ranks)
        min_heap = [(r * 1 ** 2, 1) for r in cnt.iterkeys()]
        heapq.heapify(min_heap)
        while cars > 0:
            (t, k) = heapq.heappop(min_heap)
            r = t // k ** 2
            cars -= cnt[r]
            k += 1
            heapq.heappush(min_heap, (r * k ** 2, k))
        return t