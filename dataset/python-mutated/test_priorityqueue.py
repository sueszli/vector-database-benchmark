import unittest
from reactivex.internal import PriorityQueue

class TestItem:
    __test__ = False

    def __init__(self, value, label=None):
        if False:
            print('Hello World!')
        self.value = value
        self.label = label

    def __str__(self):
        if False:
            print('Hello World!')
        if self.label:
            return '%s (%s)' % (self.value, self.label)
        else:
            return '%s' % self.value

    def __repr__(self):
        if False:
            while True:
                i = 10
        return str(self)

    def __eq__(self, other):
        if False:
            i = 10
            return i + 15
        return self.value == other.value

    def __lt__(self, other):
        if False:
            print('Hello World!')
        return self.value < other.value

    def __gt__(self, other):
        if False:
            i = 10
            return i + 15
        return self.value > other.value

class TestPriorityQueue(unittest.TestCase):

    def test_priorityqueue_count(self):
        if False:
            print('Hello World!')
        assert PriorityQueue.MIN_COUNT < 0

    def test_priorityqueue_empty(self):
        if False:
            for i in range(10):
                print('nop')
        'Must be empty on construction'
        p = PriorityQueue()
        assert len(p) == 0
        assert p.items == []
        p.enqueue(42)
        p.dequeue()
        assert len(p) == 0

    def test_priorityqueue_length(self):
        if False:
            while True:
                i = 10
        'Test that length is n after n invocations'
        p = PriorityQueue()
        assert len(p) == 0
        for n in range(42):
            p.enqueue(n)
        assert len(p) == 42
        p.dequeue()
        assert len(p) == 41
        p.remove(10)
        assert len(p) == 40
        for n in range(len(p)):
            p.dequeue()
        assert len(p) == 0

    def test_priorityqueue_enqueue_dequeue(self):
        if False:
            print('Hello World!')
        'Enqueue followed by dequeue should give the same result'
        p = PriorityQueue()
        self.assertRaises(IndexError, p.dequeue)
        p.enqueue(42)
        p.enqueue(41)
        p.enqueue(43)
        assert [p.dequeue(), p.dequeue(), p.dequeue()] == [41, 42, 43]

    def test_priorityqueue_sort_stability(self):
        if False:
            for i in range(10):
                print('nop')
        'Items with same value should be returned in the order they were\n        added'
        p = PriorityQueue()
        p.enqueue(TestItem(43, 'high'))
        p.enqueue(TestItem(42, 'first'))
        p.enqueue(TestItem(42, 'second'))
        p.enqueue(TestItem(42, 'last'))
        p.enqueue(TestItem(41, 'low'))
        assert len(p) == 5
        assert p.dequeue() == TestItem(41, 'low')
        assert p.dequeue() == TestItem(42, 'first')
        assert p.dequeue() == TestItem(42, 'second')
        assert p.dequeue() == TestItem(42, 'last')
        assert p.dequeue() == TestItem(43, 'high')

    def test_priorityqueue_remove(self):
        if False:
            print('Hello World!')
        'Remove item from queue'
        p = PriorityQueue()
        assert p.remove(42) == False
        p.enqueue(42)
        p.enqueue(41)
        p.enqueue(43)
        assert p.remove(42) == True
        assert [p.dequeue(), p.dequeue()] == [41, 43]
        p.enqueue(42)
        p.enqueue(41)
        p.enqueue(43)
        assert p.remove(41) == True
        assert [p.dequeue(), p.dequeue()] == [42, 43]
        p.enqueue(42)
        p.enqueue(41)
        p.enqueue(43)
        assert p.remove(43) == True
        assert [p.dequeue(), p.dequeue()] == [41, 42]

    def test_priorityqueue_peek(self):
        if False:
            while True:
                i = 10
        'Peek at first element in queue'
        p = PriorityQueue()
        self.assertRaises(IndexError, p.peek)
        p.enqueue(42)
        assert p.peek() == 42
        p.enqueue(41)
        assert p.peek() == 41
        p.enqueue(43)
        assert p.peek() == 41