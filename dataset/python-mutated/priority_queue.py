"""
Implementation of priority queue using linear array.
Insertion - O(n)
Extract min/max Node - O(1)
"""
import itertools

class PriorityQueueNode:

    def __init__(self, data, priority):
        if False:
            print('Hello World!')
        self.data = data
        self.priority = priority

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        return '{}: {}'.format(self.data, self.priority)

class PriorityQueue:

    def __init__(self, items=None, priorities=None):
        if False:
            for i in range(10):
                print('nop')
        'Create a priority queue with items (list or iterable).\n        If items is not passed, create empty priority queue.'
        self.priority_queue_list = []
        if items is None:
            return
        if priorities is None:
            priorities = itertools.repeat(None)
        for (item, priority) in zip(items, priorities):
            self.push(item, priority=priority)

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        return 'PriorityQueue({!r})'.format(self.priority_queue_list)

    def size(self):
        if False:
            return 10
        'Return size of the priority queue.\n        '
        return len(self.priority_queue_list)

    def push(self, item, priority=None):
        if False:
            while True:
                i = 10
        'Push the item in the priority queue.\n        if priority is not given, priority is set to the value of item.\n        '
        priority = item if priority is None else priority
        node = PriorityQueueNode(item, priority)
        for (index, current) in enumerate(self.priority_queue_list):
            if current.priority < node.priority:
                self.priority_queue_list.insert(index, node)
                return
        self.priority_queue_list.append(node)

    def pop(self):
        if False:
            while True:
                i = 10
        'Remove and return the item with the lowest priority.\n        '
        return self.priority_queue_list.pop().data