import heapq

class Solution(object):

    def maxAverageRatio(self, classes, extraStudents):
        if False:
            while True:
                i = 10
        '\n        :type classes: List[List[int]]\n        :type extraStudents: int\n        :rtype: float\n        '

        def profit(a, b):
            if False:
                for i in range(10):
                    print('nop')
            return float(a + 1) / (b + 1) - float(a) / b
        max_heap = [(-profit(a, b), a, b) for (a, b) in classes]
        heapq.heapify(max_heap)
        while extraStudents:
            (v, a, b) = heapq.heappop(max_heap)
            (a, b) = (a + 1, b + 1)
            heapq.heappush(max_heap, (-profit(a, b), a, b))
            extraStudents -= 1
        return sum((float(a) / b for (v, a, b) in max_heap)) / len(classes)