import heapq
from sys import maxsize
from typing import Generic, List, Tuple, TypeVar
_T1 = TypeVar('_T1')

class PriorityQueue(Generic[_T1]):
    """Priority queue for scheduling. Note that methods aren't thread-safe."""
    MIN_COUNT = ~maxsize

    def __init__(self) -> None:
        if False:
            while True:
                i = 10
        self.items: List[Tuple[_T1, int]] = []
        self.count = PriorityQueue.MIN_COUNT

    def __len__(self) -> int:
        if False:
            while True:
                i = 10
        'Returns length of queue'
        return len(self.items)

    def peek(self) -> _T1:
        if False:
            while True:
                i = 10
        'Returns first item in queue without removing it'
        return self.items[0][0]

    def dequeue(self) -> _T1:
        if False:
            return 10
        'Returns and removes item with lowest priority from queue'
        item: _T1 = heapq.heappop(self.items)[0]
        if not self.items:
            self.count = PriorityQueue.MIN_COUNT
        return item

    def enqueue(self, item: _T1) -> None:
        if False:
            while True:
                i = 10
        'Adds item to queue'
        heapq.heappush(self.items, (item, self.count))
        self.count += 1

    def remove(self, item: _T1) -> bool:
        if False:
            return 10
        'Remove given item from queue'
        for (index, _item) in enumerate(self.items):
            if _item[0] == item:
                self.items.pop(index)
                heapq.heapify(self.items)
                return True
        return False

    def clear(self) -> None:
        if False:
            while True:
                i = 10
        'Remove all items from the queue.'
        self.items = []
        self.count = PriorityQueue.MIN_COUNT