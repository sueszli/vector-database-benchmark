class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.next = next

class Solution(object):

    def deleteNodes(self, head, m, n):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type head: ListNode\n        :type m: int\n        :type n: int\n        :rtype: ListNode\n        '
        head = dummy = ListNode(next=head)
        while head:
            for _ in xrange(m):
                if not head.next:
                    return dummy.next
                head = head.next
            prev = head
            for _ in xrange(n):
                if not head.next:
                    prev.next = None
                    return dummy.next
                head = head.next
            prev.next = head.next
        return dummy.next