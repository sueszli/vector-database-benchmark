import heapq

class Solution(object):

    def minAvailableDuration(self, slots1, slots2, duration):
        if False:
            return 10
        '\n        :type slots1: List[List[int]]\n        :type slots2: List[List[int]]\n        :type duration: int\n        :rtype: List[int]\n        '
        min_heap = list(filter(lambda slot: slot[1] - slot[0] >= duration, slots1 + slots2))
        heapq.heapify(min_heap)
        while len(min_heap) > 1:
            left = heapq.heappop(min_heap)
            right = min_heap[0]
            if left[1] - right[0] >= duration:
                return [right[0], right[0] + duration]
        return []

class Solution2(object):

    def minAvailableDuration(self, slots1, slots2, duration):
        if False:
            return 10
        '\n        :type slots1: List[List[int]]\n        :type slots2: List[List[int]]\n        :type duration: int\n        :rtype: List[int]\n        '
        slots1.sort(key=lambda x: x[0])
        slots2.sort(key=lambda x: x[0])
        (i, j) = (0, 0)
        while i < len(slots1) and j < len(slots2):
            left = max(slots1[i][0], slots2[j][0])
            right = min(slots1[i][1], slots2[j][1])
            if left + duration <= right:
                return [left, left + duration]
            if slots1[i][1] < slots2[j][1]:
                i += 1
            else:
                j += 1
        return []