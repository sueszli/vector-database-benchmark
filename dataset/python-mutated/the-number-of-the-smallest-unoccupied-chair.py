import heapq

class Solution(object):

    def smallestChair(self, times, targetFriend):
        if False:
            print('Hello World!')
        '\n        :type times: List[List[int]]\n        :type targetFriend: int\n        :rtype: int\n        '
        events = []
        for (i, (s, e)) in enumerate(times):
            events.append((s, True, i))
            events.append((e, False, i))
        events.sort()
        lookup = {}
        min_heap = []
        for (_, arrival, i) in events:
            if not arrival:
                heapq.heappush(min_heap, lookup.pop(i))
                continue
            lookup[i] = heapq.heappop(min_heap) if min_heap else len(lookup)
            if i == targetFriend:
                break
        return lookup[targetFriend]