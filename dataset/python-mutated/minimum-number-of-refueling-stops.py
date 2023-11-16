import heapq

class Solution(object):

    def minRefuelStops(self, target, startFuel, stations):
        if False:
            print('Hello World!')
        '\n        :type target: int\n        :type startFuel: int\n        :type stations: List[List[int]]\n        :rtype: int\n        '
        max_heap = []
        stations.append((target, float('inf')))
        result = prev = 0
        for (location, capacity) in stations:
            startFuel -= location - prev
            while max_heap and startFuel < 0:
                startFuel += -heapq.heappop(max_heap)
                result += 1
            if startFuel < 0:
                return -1
            heapq.heappush(max_heap, -capacity)
            prev = location
        return result