class ListNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            print('Hello World!')
        if self:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def rotateRight(self, head, k):
        if False:
            i = 10
            return i + 15
        '\n        :type head: ListNode\n        :type k: int\n        :rtype: ListNode\n        '
        if not head or not head.next:
            return head
        (n, cur) = (1, head)
        while cur.next:
            cur = cur.next
            n += 1
        cur.next = head
        (cur, tail) = (head, cur)
        for _ in xrange(n - k % n):
            tail = cur
            cur = cur.next
        tail.next = None
        return cur