"""
Running Median

Compute the running median of a sequence of numbers.
That is, given a stream of numbers, print out the median of the list so far on each new element.
Recall that the median of an even-numbered list is the average of the two middle numbers.

Input: [2, 1, 5, 7, 2, 0, 5]
Output:
2
1.5
2
3.5
2
2
2

=========================================
Using 2 heaps (max and min Priority Queues) balance the left and right side of the stream.
    Time Complexity:    O(N LogN)
    Space Complexity:   O(N)
"""
import heapq

class PriorityQueue:

    def __init__(self, is_min=True):
        if False:
            for i in range(10):
                print('nop')
        self.data = []
        self.is_min = is_min

    def push(self, el):
        if False:
            print('Hello World!')
        if not self.is_min:
            el = -el
        heapq.heappush(self.data, el)

    def pop(self):
        if False:
            i = 10
            return i + 15
        el = heapq.heappop(self.data)
        if not self.is_min:
            el = -el
        return el

    def peek(self):
        if False:
            i = 10
            return i + 15
        el = self.data[0]
        if not self.is_min:
            el = -el
        return el

    def count(self):
        if False:
            for i in range(10):
                print('nop')
        return len(self.data)

def running_median(stream):
    if False:
        return 10
    left_heap = PriorityQueue(False)
    right_heap = PriorityQueue()
    for number in stream:
        if left_heap.count() == 0:
            left_heap.push(number)
        elif left_heap.count() > right_heap.count():
            if left_heap.peek() > number:
                right_heap.push(left_heap.pop())
                left_heap.push(number)
            else:
                right_heap.push(number)
        elif right_heap.peek() < number:
            left_heap.push(right_heap.pop())
            right_heap.push(number)
        else:
            left_heap.push(number)
        if left_heap.count() > right_heap.count():
            print(left_heap.peek())
        else:
            print((left_heap.peek() + right_heap.peek()) / 2)
running_median([2, 1, 5, 7, 2, 0, 5])