class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

class Solution(object):

    def removeElements(self, head, val):
        if False:
            print('Hello World!')
        dummy = ListNode(float('-inf'))
        dummy.next = head
        (prev, curr) = (dummy, dummy.next)
        while curr:
            if curr.val == val:
                prev.next = curr.next
            else:
                prev = curr
            curr = curr.next
        return dummy.next