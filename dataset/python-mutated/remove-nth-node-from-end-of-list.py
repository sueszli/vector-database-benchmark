class ListNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            while True:
                i = 10
        if self is None:
            return 'Nil'
        else:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def removeNthFromEnd(self, head, n):
        if False:
            return 10
        dummy = ListNode(-1)
        dummy.next = head
        (slow, fast) = (dummy, dummy)
        for i in xrange(n):
            fast = fast.next
        while fast.next:
            (slow, fast) = (slow.next, fast.next)
        slow.next = slow.next.next
        return dummy.next