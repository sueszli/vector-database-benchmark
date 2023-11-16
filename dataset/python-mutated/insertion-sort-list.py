class ListNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            return 10
        if self:
            return '{} -> {}'.format(self.val, repr(self.next))
        else:
            return 'Nil'

class Solution(object):

    def insertionSortList(self, head):
        if False:
            print('Hello World!')
        if head is None or self.isSorted(head):
            return head
        dummy = ListNode(-2147483648)
        dummy.next = head
        (cur, sorted_tail) = (head.next, head)
        while cur:
            prev = dummy
            while prev.next.val < cur.val:
                prev = prev.next
            if prev == sorted_tail:
                (cur, sorted_tail) = (cur.next, cur)
            else:
                (cur.next, prev.next, sorted_tail.next) = (prev.next, cur, cur.next)
                cur = sorted_tail.next
        return dummy.next

    def isSorted(self, head):
        if False:
            return 10
        while head and head.next:
            if head.val > head.next.val:
                return False
            head = head.next
        return True