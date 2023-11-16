class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.next = next

class Solution(object):

    def sortLinkedList(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: Optional[ListNode]\n        :rtype: Optional[ListNode]\n        '
        (tail, curr, head.next) = (head, head.next, None)
        while curr:
            if curr.val > 0:
                (curr.next, tail.next, tail, curr) = (None, curr, curr, curr.next)
            else:
                (curr.next, head, curr) = (head, curr, curr.next)
        return head