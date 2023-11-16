class MinHeap:

    def __init__(self):
        if False:
            return 10
        self.min_heap = []

    def peek(self) -> int:
        if False:
            print('Hello World!')
        if not self.min_heap:
            return None
        return self.min_heap[0]

    def push(self, val: int):
        if False:
            i = 10
            return i + 15
        self.min_heap.append(val)
        size = len(self.min_heap)
        self.__shift_up(size - 1)

    def __shift_up(self, i: int):
        if False:
            return 10
        while (i - 1) // 2 >= 0 and self.min_heap[i] < self.min_heap[(i - 1) // 2]:
            (self.min_heap[i], self.min_heap[(i - 1) // 2]) = (self.min_heap[(i - 1) // 2], self.min_heap[i])
            i = (i - 1) // 2

    def pop(self) -> int:
        if False:
            for i in range(10):
                print('nop')
        if not self.min_heap:
            raise IndexError('堆为空')
        size = len(self.min_heap)
        (self.min_heap[0], self.min_heap[size - 1]) = (self.min_heap[size - 1], self.min_heap[0])
        val = self.min_heap.pop()
        size -= 1
        self.__shift_down(0, size)
        return val

    def __shift_down(self, i: int, n: int):
        if False:
            for i in range(10):
                print('nop')
        while 2 * i + 1 < n:
            (left, right) = (2 * i + 1, 2 * i + 2)
            if 2 * i + 2 >= n:
                larger = left
            elif self.min_heap[left] <= self.min_heap[right]:
                larger = left
            else:
                larger = right
            if self.min_heap[i] > self.min_heap[larger]:
                (self.min_heap[i], self.min_heap[larger]) = (self.min_heap[larger], self.min_heap[i])
                i = larger
            else:
                break

    def __buildMinHeap(self, nums: [int]):
        if False:
            for i in range(10):
                print('nop')
        size = len(nums)
        for i in range(size):
            self.min_heap.append(nums[i])
        for i in range((size - 2) // 2, -1, -1):
            self.__shift_down(i, size)

    def minHeapSort(self, nums: [int]) -> [int]:
        if False:
            i = 10
            return i + 15
        self.__buildMinHeap(nums)
        size = len(self.min_heap)
        for i in range(size - 1, -1, -1):
            (self.min_heap[0], self.min_heap[i]) = (self.min_heap[i], self.min_heap[0])
            self.__shift_down(0, i)
        return self.min_heap

class Solution:

    def minHeapSort(self, nums: [int]) -> [int]:
        if False:
            return 10
        return MinHeap().minHeapSort(nums)

    def sortArray(self, nums: [int]) -> [int]:
        if False:
            return 10
        return self.minHeapSort(nums)
print(Solution().sortArray([10, 25, 6, 8, 7, 1, 20, 23, 16, 19, 17, 3, 18, 14]))