class Node(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.next = None
        self.random = None

class Solution(object):

    def copyRandomList(self, head):
        if False:
            while True:
                i = 10
        current = head
        while current:
            copied = Node(current.val)
            copied.next = current.next
            current.next = copied
            current = copied.next
        current = head
        while current:
            if current.random:
                current.next.random = current.random.next
            current = current.next.next
        dummy = Node(0)
        (copied_current, current) = (dummy, head)
        while current:
            copied_current.next = current.next
            current.next = current.next.next
            (copied_current, current) = (copied_current.next, current.next)
        return dummy.next

class Solution2(object):

    def copyRandomList(self, head):
        if False:
            while True:
                i = 10
        dummy = Node(0)
        (current, prev, copies) = (head, dummy, {})
        while current:
            copied = Node(current.val)
            copies[current] = copied
            prev.next = copied
            (prev, current) = (prev.next, current.next)
        current = head
        while current:
            if current.random:
                copies[current].random = copies[current.random]
            current = current.next
        return dummy.next
from collections import defaultdict

class Solution3(object):

    def copyRandomList(self, head):
        if False:
            print('Hello World!')
        '\n        :type head: RandomListNode\n        :rtype: RandomListNode\n        '
        clone = defaultdict(lambda : Node(0))
        clone[None] = None
        cur = head
        while cur:
            clone[cur].val = cur.val
            clone[cur].next = clone[cur.next]
            clone[cur].random = clone[cur.random]
            cur = cur.next
        return clone[head]