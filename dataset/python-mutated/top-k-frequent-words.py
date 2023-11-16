import collections
import heapq
from random import randint

class Solution(object):

    def topKFrequent(self, words, k):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type words: List[str]\n        :type k: int\n        :rtype: List[str]\n        '
        counts = collections.Counter(words)
        p = []
        for (key, val) in counts.iteritems():
            p.append((-val, key))
        self.kthElement(p, k - 1)
        result = []
        sorted_p = sorted(p[:k])
        for i in xrange(k):
            result.append(sorted_p[i][1])
        return result

    def kthElement(self, nums, k):
        if False:
            print('Hello World!')

        def PartitionAroundPivot(left, right, pivot_idx, nums):
            if False:
                while True:
                    i = 10
            pivot_value = nums[pivot_idx]
            new_pivot_idx = left
            (nums[pivot_idx], nums[right]) = (nums[right], nums[pivot_idx])
            for i in xrange(left, right):
                if nums[i] < pivot_value:
                    (nums[i], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[i])
                    new_pivot_idx += 1
            (nums[right], nums[new_pivot_idx]) = (nums[new_pivot_idx], nums[right])
            return new_pivot_idx
        (left, right) = (0, len(nums) - 1)
        while left <= right:
            pivot_idx = randint(left, right)
            new_pivot_idx = PartitionAroundPivot(left, right, pivot_idx, nums)
            if new_pivot_idx == k:
                return
            elif new_pivot_idx > k:
                right = new_pivot_idx - 1
            else:
                left = new_pivot_idx + 1

class Solution2(object):

    def topKFrequent(self, words, k):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :type k: int\n        :rtype: List[str]\n        '

        class MinHeapObj(object):

            def __init__(self, val):
                if False:
                    print('Hello World!')
                self.val = val

            def __lt__(self, other):
                if False:
                    return 10
                return self.val[1] > other.val[1] if self.val[0] == other.val[0] else self.val < other.val

            def __eq__(self, other):
                if False:
                    return 10
                return self.val == other.val

            def __str__(self):
                if False:
                    while True:
                        i = 10
                return str(self.val)
        counts = collections.Counter(words)
        min_heap = []
        for (word, count) in counts.iteritems():
            heapq.heappush(min_heap, MinHeapObj((count, word)))
            if len(min_heap) == k + 1:
                heapq.heappop(min_heap)
        result = []
        while min_heap:
            result.append(heapq.heappop(min_heap).val[1])
        return result[::-1]

class Solution3(object):

    def topKFrequent(self, words, k):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :type k: int\n        :rtype: List[str]\n        '
        counts = collections.Counter(words)
        buckets = [[] for _ in xrange(len(words) + 1)]
        for (word, count) in counts.iteritems():
            buckets[count].append(word)
        pairs = []
        for i in reversed(xrange(len(words))):
            for word in buckets[i]:
                pairs.append((-i, word))
            if len(pairs) >= k:
                break
        pairs.sort()
        return [pair[1] for pair in pairs[:k]]
from collections import Counter

class Solution4(object):

    def topKFrequent(self, words, k):
        if False:
            i = 10
            return i + 15
        '\n        :type words: List[str]\n        :type k: int\n        :rtype: List[str]\n        '
        counter = Counter(words)
        candidates = counter.keys()
        candidates.sort(key=lambda w: (-counter[w], w))
        return candidates[:k]