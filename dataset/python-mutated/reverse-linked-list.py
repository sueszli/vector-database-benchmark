class ListNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            return 10
        if self:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def reverseList(self, head):
        if False:
            while True:
                i = 10
        dummy = ListNode(float('-inf'))
        while head:
            (dummy.next, head.next, head) = (head, dummy.next, head.next)
        return dummy.next

class Solution2(object):

    def reverseList(self, head):
        if False:
            while True:
                i = 10
        [begin, end] = self.reverseListRecu(head)
        return begin

    def reverseListRecu(self, head):
        if False:
            while True:
                i = 10
        if not head:
            return [None, None]
        [begin, end] = self.reverseListRecu(head.next)
        if end:
            end.next = head
            head.next = None
            return [begin, head]
        else:
            return [head, head]