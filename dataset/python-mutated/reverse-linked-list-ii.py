class ListNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            return 10
        if self:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def reverseBetween(self, head, m, n):
        if False:
            return 10
        (diff, dummy, cur) = (n - m + 1, ListNode(-1), head)
        dummy.next = head
        last_unswapped = dummy
        while cur and m > 1:
            (cur, last_unswapped, m) = (cur.next, cur, m - 1)
        (prev, first_swapped) = (last_unswapped, cur)
        while cur and diff > 0:
            (cur.next, prev, cur, diff) = (prev, cur, cur.next, diff - 1)
        (last_unswapped.next, first_swapped.next) = (prev, cur)
        return dummy.next