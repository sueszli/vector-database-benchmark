class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            return 10
        if self is None:
            return 'Nil'
        else:
            return '{} -> {}'.format(self.val, repr(self.next))

class Solution(object):

    def deleteDuplicates(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        dummy = ListNode(0)
        (pre, cur) = (dummy, head)
        while cur:
            if cur.next and cur.next.val == cur.val:
                val = cur.val
                while cur and cur.val == val:
                    cur = cur.next
                pre.next = cur
            else:
                pre.next = cur
                pre = cur
                cur = cur.next
        return dummy.next