import collections

class Solution(object):

    def minimumCost(self, start, target, specialRoads):
        if False:
            while True:
                i = 10
        '\n        :type start: List[int]\n        :type target: List[int]\n        :type specialRoads: List[List[int]]\n        :rtype: int\n        '
        (start, target) = (tuple(start), tuple(target))
        adj = collections.defaultdict(list, {target: []})
        for (x1, y1, x2, y2, c) in specialRoads:
            adj[x1, y1].append((x2, y2, c))
        dist = {start: 0}
        lookup = set()
        while len(lookup) != len(dist):
            (d, x1, y1) = min(((dist[x1, y1], x1, y1) for (x1, y1) in dist.iterkeys() if (x1, y1) not in lookup))
            lookup.add((x1, y1))
            if (x1, y1) == target:
                return d
            for (x2, y2, c) in adj[x1, y1]:
                if not ((x2, y2) not in dist or dist[x2, y2] > d + c):
                    continue
                dist[x2, y2] = d + c
            for (x2, y2) in adj.iterkeys():
                if not ((x2, y2) not in dist or dist[x2, y2] > d + abs(x2 - x1) + abs(y2 - y1)):
                    continue
                dist[x2, y2] = d + abs(x2 - x1) + abs(y2 - y1)
import collections
import heapq

class Solution2(object):

    def minimumCost(self, start, target, specialRoads):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type start: List[int]\n        :type target: List[int]\n        :type specialRoads: List[List[int]]\n        :rtype: int\n        '
        (start, target) = (tuple(start), tuple(target))
        adj = collections.defaultdict(list, {target: []})
        for (x1, y1, x2, y2, c) in specialRoads:
            adj[x1, y1].append((x2, y2, c))
        min_heap = [(0, start)]
        dist = {start: 0}
        while min_heap:
            (d, (x1, y1)) = heapq.heappop(min_heap)
            if d > dist[x1, y1]:
                continue
            if (x1, y1) == target:
                return d
            for (x2, y2, c) in adj[x1, y1]:
                if not ((x2, y2) not in dist or dist[x2, y2] > d + c):
                    continue
                dist[x2, y2] = d + c
                heapq.heappush(min_heap, (dist[x2, y2], (x2, y2)))
            for (x2, y2) in adj.iterkeys():
                if not ((x2, y2) not in dist or dist[x2, y2] > d + abs(x2 - x1) + abs(y2 - y1)):
                    continue
                dist[x2, y2] = d + abs(x2 - x1) + abs(y2 - y1)
                heapq.heappush(min_heap, (dist[x2, y2], (x2, y2)))
        return -1