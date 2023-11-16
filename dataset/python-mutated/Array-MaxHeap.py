class MaxHeap:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.max_heap = []

    def peek(self) -> int:
        if False:
            i = 10
            return i + 15
        if not self.max_heap:
            return None
        return self.max_heap[0]

    def push(self, val: int):
        if False:
            for i in range(10):
                print('nop')
        self.max_heap.append(val)
        size = len(self.max_heap)
        self.__shift_up(size - 1)

    def __shift_up(self, i: int):
        if False:
            for i in range(10):
                print('nop')
        while (i - 1) // 2 >= 0 and self.max_heap[i] > self.max_heap[(i - 1) // 2]:
            (self.max_heap[i], self.max_heap[(i - 1) // 2]) = (self.max_heap[(i - 1) // 2], self.max_heap[i])
            i = (i - 1) // 2

    def pop(self) -> int:
        if False:
            while True:
                i = 10
        if not self.max_heap:
            raise IndexError('堆为空')
        size = len(self.max_heap)
        (self.max_heap[0], self.max_heap[size - 1]) = (self.max_heap[size - 1], self.max_heap[0])
        val = self.max_heap.pop()
        size -= 1
        self.__shift_down(0, size)
        return val

    def __shift_down(self, i: int, n: int):
        if False:
            i = 10
            return i + 15
        while 2 * i + 1 < n:
            (left, right) = (2 * i + 1, 2 * i + 2)
            if 2 * i + 2 >= n:
                larger = left
            elif self.max_heap[left] >= self.max_heap[right]:
                larger = left
            else:
                larger = right
            if self.max_heap[i] < self.max_heap[larger]:
                (self.max_heap[i], self.max_heap[larger]) = (self.max_heap[larger], self.max_heap[i])
                i = larger
            else:
                break

class Solution:

    def maxHeapOperations(self):
        if False:
            while True:
                i = 10
        max_heap = MaxHeap()
        max_heap.push(3)
        print(max_heap.peek())
        max_heap.push(2)
        print(max_heap.peek())
        max_heap.push(4)
        print(max_heap.peek())
        max_heap.pop()
        print(max_heap.peek())
print(Solution().maxHeapOperations())