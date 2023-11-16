class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def partition(self, head, x):
        if False:
            i = 10
            return i + 15
        (dummySmaller, dummyGreater) = (ListNode(-1), ListNode(-1))
        (smaller, greater) = (dummySmaller, dummyGreater)
        while head:
            if head.val < x:
                smaller.next = head
                smaller = smaller.next
            else:
                greater.next = head
                greater = greater.next
            head = head.next
        smaller.next = dummyGreater.next
        greater.next = None
        return dummySmaller.next