class ListNode(object):

    def __init__(self, x):
        if False:
            print('Hello World!')
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            for i in range(10):
                print('nop')
        if self:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def reverseKGroup(self, head, k):
        if False:
            while True:
                i = 10
        dummy = ListNode(-1)
        dummy.next = head
        (cur, cur_dummy) = (head, dummy)
        length = 0
        while cur:
            next_cur = cur.next
            length = (length + 1) % k
            if length == 0:
                next_dummy = cur_dummy.next
                self.reverse(cur_dummy, cur.next)
                cur_dummy = next_dummy
            cur = next_cur
        return dummy.next

    def reverse(self, begin, end):
        if False:
            for i in range(10):
                print('nop')
        first = begin.next
        cur = first.next
        while cur != end:
            first.next = cur.next
            cur.next = begin.next
            begin.next = cur
            cur = first.next