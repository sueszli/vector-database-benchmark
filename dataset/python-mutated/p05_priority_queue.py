"""
Topic: 优先级队列
Desc : 
"""
import heapq

class PriorityQueue:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self._queue = []
        self._index = 0

    def push(self, item, priority):
        if False:
            while True:
                i = 10
        heapq.heappush(self._queue, (-priority, self._index, item))
        self._index += 1

    def pop(self):
        if False:
            for i in range(10):
                print('nop')
        return heapq.heappop(self._queue)[-1]