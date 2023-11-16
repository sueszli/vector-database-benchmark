from sortedcontainers import SortedList

class Solution(object):

    def medianSlidingWindow(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[float]\n        '
        sl = SortedList((float(nums[i]) for i in xrange(k)))
        result = [(sl[k // 2] + sl[k // 2 - (1 - k % 2)]) / 2]
        for i in xrange(k, len(nums)):
            sl.add(float(nums[i]))
            sl.remove(nums[i - k])
            result.append((sl[k // 2] + sl[k // 2 - (1 - k % 2)]) / 2)
        return result
import collections
import heapq

class Solution2(object):

    def medianSlidingWindow(self, nums, k):
        if False:
            print('Hello World!')
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[float]\n        '

        def lazy_delete(heap, to_remove, sign):
            if False:
                while True:
                    i = 10
            while heap and sign * heap[0] in to_remove:
                to_remove[sign * heap[0]] -= 1
                if not to_remove[sign * heap[0]]:
                    del to_remove[sign * heap[0]]
                heapq.heappop(heap)

        def full_delete(heap, to_remove, sign):
            if False:
                for i in range(10):
                    print('nop')
            result = []
            for x in heap:
                if sign * x not in to_remove:
                    result.append(x)
                    continue
                to_remove[sign * x] -= 1
                if not to_remove[sign * x]:
                    del to_remove[sign * x]
            heap[:] = result
            heapify(heap)
        (min_heap, max_heap) = ([], [])
        for i in xrange(k):
            if i % 2 == 0:
                heapq.heappush(min_heap, -heapq.heappushpop(max_heap, -nums[i]))
            else:
                heapq.heappush(max_heap, -heapq.heappushpop(min_heap, nums[i]))
        result = [float(min_heap[0])] if k % 2 else [(min_heap[0] - max_heap[0]) / 2.0]
        to_remove = collections.defaultdict(int)
        for i in xrange(k, len(nums)):
            heapq.heappush(max_heap, -heapq.heappushpop(min_heap, nums[i]))
            if nums[i - k] > -max_heap[0]:
                heapq.heappush(min_heap, -heapq.heappop(max_heap))
            to_remove[nums[i - k]] += 1
            lazy_delete(max_heap, to_remove, -1)
            lazy_delete(min_heap, to_remove, 1)
            if len(min_heap) + len(max_heap) > 2 * k:
                full_delete(max_heap, to_remove, -1)
                full_delete(min_heap, to_remove, 1)
            result.append(float(min_heap[0]) if k % 2 else (min_heap[0] - max_heap[0]) / 2.0)
        return result
import collections
import heapq

class Solution3(object):

    def medianSlidingWindow(self, nums, k):
        if False:
            while True:
                i = 10
        '\n        :type nums: List[int]\n        :type k: int\n        :rtype: List[float]\n        '

        def lazy_delete(heap, to_remove, sign):
            if False:
                return 10
            while heap and sign * heap[0] in to_remove:
                to_remove[sign * heap[0]] -= 1
                if not to_remove[sign * heap[0]]:
                    del to_remove[sign * heap[0]]
                heapq.heappop(heap)
        (min_heap, max_heap) = ([], [])
        for i in xrange(k):
            if i % 2 == 0:
                heapq.heappush(min_heap, -heapq.heappushpop(max_heap, -nums[i]))
            else:
                heapq.heappush(max_heap, -heapq.heappushpop(min_heap, nums[i]))
        result = [float(min_heap[0])] if k % 2 else [(min_heap[0] - max_heap[0]) / 2.0]
        to_remove = collections.defaultdict(int)
        for i in xrange(k, len(nums)):
            heapq.heappush(max_heap, -heapq.heappushpop(min_heap, nums[i]))
            if nums[i - k] > -max_heap[0]:
                heapq.heappush(min_heap, -heapq.heappop(max_heap))
            to_remove[nums[i - k]] += 1
            lazy_delete(max_heap, to_remove, -1)
            lazy_delete(min_heap, to_remove, 1)
            result.append(float(min_heap[0]) if k % 2 else (min_heap[0] - max_heap[0]) / 2.0)
        return result