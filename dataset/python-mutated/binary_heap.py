"""
Binary Heap. A min heap is a complete binary tree where each node is smaller than
its children. The root, therefore, is the minimum element in the tree. The min
heap uses an array to represent the data and operation. For example a min heap:

     4
   /   \\
  50    7
 / \\   /
55 90 87

Heap [0, 4, 50, 7, 55, 90, 87]

Method in class: insert, remove_min
For example insert(2) in a min heap:

     4                     4                     2
   /   \\                 /   \\                 /   \\
  50    7      -->     50     2       -->     50    4
 / \\   /  \\           /  \\   / \\             /  \\  /  \\
55 90 87   2         55  90 87  7           55  90 87  7

For example remove_min() in a min heap:

     4                     87                    7
   /   \\                 /   \\                 /   \\
  50    7      -->     50     7       -->     50    87
 / \\   /              /  \\                   /  \\
55 90 87             55  90                 55  90

"""
from abc import ABCMeta, abstractmethod

class AbstractHeap(metaclass=ABCMeta):
    """Abstract Class for Binary Heap."""

    def __init__(self):
        if False:
            while True:
                i = 10
        'Pass.'

    @abstractmethod
    def perc_up(self, i):
        if False:
            i = 10
            return i + 15
        'Pass.'

    @abstractmethod
    def insert(self, val):
        if False:
            i = 10
            return i + 15
        'Pass.'

    @abstractmethod
    def perc_down(self, i):
        if False:
            return 10
        'Pass.'

    @abstractmethod
    def min_child(self, i):
        if False:
            return 10
        'Pass.'

    @abstractmethod
    def remove_min(self):
        if False:
            return 10
        'Pass.'

class BinaryHeap(AbstractHeap):
    """Binary Heap Class"""

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.current_size = 0
        self.heap = [0]

    def perc_up(self, i):
        if False:
            for i in range(10):
                print('nop')
        while i // 2 > 0:
            if self.heap[i] < self.heap[i // 2]:
                (self.heap[i], self.heap[i // 2]) = (self.heap[i // 2], self.heap[i])
            i = i // 2

    def insert(self, val):
        if False:
            while True:
                i = 10
        '\n        Method insert always start by inserting the element at the bottom.\n        It inserts rightmost spot so as to maintain the complete tree property.\n        Then, it fixes the tree by swapping the new element with its parent,\n        until it finds an appropriate spot for the element. It essentially\n        perc_up the minimum element\n        Complexity: O(logN)\n        '
        self.heap.append(val)
        self.current_size = self.current_size + 1
        self.perc_up(self.current_size)
        '\n        Method min_child returns the index of smaller of 2 children of parent at index i\n        '

    def min_child(self, i):
        if False:
            for i in range(10):
                print('nop')
        if 2 * i + 1 > self.current_size:
            return 2 * i
        if self.heap[2 * i] > self.heap[2 * i + 1]:
            return 2 * i + 1
        return 2 * i

    def perc_down(self, i):
        if False:
            i = 10
            return i + 15
        while 2 * i < self.current_size:
            min_child = self.min_child(i)
            if self.heap[min_child] < self.heap[i]:
                (self.heap[min_child], self.heap[i]) = (self.heap[i], self.heap[min_child])
            i = min_child
    '\n        Remove Min method removes the minimum element and swap it with the last\n        element in the heap( the bottommost, rightmost element). Then, it\n        perc_down this element, swapping it with one of its children until the\n        min heap property is restored\n        Complexity: O(logN)\n    '

    def remove_min(self):
        if False:
            i = 10
            return i + 15
        ret = self.heap[1]
        self.heap[1] = self.heap[self.current_size]
        self.current_size = self.current_size - 1
        self.heap.pop()
        self.perc_down(1)
        return ret