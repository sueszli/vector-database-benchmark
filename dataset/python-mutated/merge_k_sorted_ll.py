"""
Merge K Sorted Linked Lists

Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.

=========================================
Using Priority Queue (heap) in each step chose the smallest element from the lists and add it to the result list.
    Time Complexity: 	O(N * LogK)  , LogK is for adding and deleting from Priority queue
    Space Complexity: 	O(N)
Using Divide and Conquer, similar to Merge sort.
    Time Complexity:    O(N * LogK)
    Space Complexity:   O(1)  , (using the same old list)
"""
from ll_helpers import ListNode
import heapq

class PQNode:

    def __init__(self, node):
        if False:
            for i in range(10):
                print('nop')
        self.val = node.val
        self.node = node

    def __lt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.val < other.val

class PriorityQueue:

    def __init__(self):
        if False:
            while True:
                i = 10
        self.data = []

    def push(self, node):
        if False:
            i = 10
            return i + 15
        heapq.heappush(self.data, PQNode(node))

    def pop(self):
        if False:
            i = 10
            return i + 15
        return heapq.heappop(self.data).node

    def is_empty(self):
        if False:
            return 10
        return len(self.data) == 0

def merge_k_lists_1(lists):
    if False:
        for i in range(10):
            print('nop')
    heap = PriorityQueue()
    for node in lists:
        if node is not None:
            heap.push(node)
    result = ListNode(-1)
    pointer = result
    while not heap.is_empty():
        node = heap.pop()
        pointer.next = node
        pointer = pointer.next
        node = node.next
        if node is not None:
            heap.push(node)
    return result.next

def merge_k_lists_2(lists):
    if False:
        print('Hello World!')
    n = len(lists)
    if n == 0:
        return None
    step = 1
    while step < n:
        i = 0
        while i + step < n:
            lists[i] = merge_2_lists(lists[i], lists[i + step])
            i += 2 * step
        step *= 2
    return lists[0]

def merge_2_lists(l1, l2):
    if False:
        i = 10
        return i + 15
    result = ListNode(-1)
    pointer = result
    while l1 is not None and l2 is not None:
        if l1.val < l2.val:
            pointer.next = l1
            l1 = l1.next
        else:
            pointer.next = l2
            l2 = l2.next
        pointer = pointer.next
    if l1 is not None:
        pointer.next = l1
    if l2 is not None:
        pointer.next = l2
    return result.next