import itertools
import heapq

class Solution(object):

    def busiestServers(self, k, arrival, load):
        if False:
            print('Hello World!')
        '\n        :type k: int\n        :type arrival: List[int]\n        :type load: List[int]\n        :rtype: List[int]\n        '
        count = [0] * k
        min_heap_of_endtimes = []
        min_heap_of_nodes_after_curr = []
        min_heap_of_nodes_before_curr = range(k)
        for (i, (t, l)) in enumerate(itertools.izip(arrival, load)):
            if i % k == 0:
                (min_heap_of_nodes_before_curr, min_heap_of_nodes_after_curr) = ([], min_heap_of_nodes_before_curr)
            while min_heap_of_endtimes and min_heap_of_endtimes[0][0] <= t:
                (_, free) = heapq.heappop(min_heap_of_endtimes)
                if free < i % k:
                    heapq.heappush(min_heap_of_nodes_before_curr, free)
                else:
                    heapq.heappush(min_heap_of_nodes_after_curr, free)
            min_heap_of_candidates = min_heap_of_nodes_after_curr if min_heap_of_nodes_after_curr else min_heap_of_nodes_before_curr
            if not min_heap_of_candidates:
                continue
            node = heapq.heappop(min_heap_of_candidates)
            count[node] += 1
            heapq.heappush(min_heap_of_endtimes, (t + l, node))
        max_count = max(count)
        return [i for i in xrange(k) if count[i] == max_count]
import sortedcontainers
import itertools
import heapq

class Solution2(object):

    def busiestServers(self, k, arrival, load):
        if False:
            i = 10
            return i + 15
        '\n        :type k: int\n        :type arrival: List[int]\n        :type load: List[int]\n        :rtype: List[int]\n        '
        count = [0] * k
        min_heap_of_endtimes = []
        availables = sortedcontainers.SortedList(xrange(k))
        for (i, (t, l)) in enumerate(itertools.izip(arrival, load)):
            while min_heap_of_endtimes and min_heap_of_endtimes[0][0] <= t:
                (_, free) = heapq.heappop(min_heap_of_endtimes)
                availables.add(free)
            if not availables:
                continue
            idx = availables.bisect_left(i % k) % len(availables)
            node = availables.pop(idx)
            count[node] += 1
            heapq.heappush(min_heap_of_endtimes, (t + l, node))
        max_count = max(count)
        return [i for i in xrange(k) if count[i] == max_count]