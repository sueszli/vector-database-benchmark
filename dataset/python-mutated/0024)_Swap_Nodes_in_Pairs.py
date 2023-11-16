class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.next = next

class Solution:

    def swapPairs(self, head: ListNode) -> ListNode:
        if False:
            while True:
                i = 10
        prev = dummy = ListNode(None)
        while head and head.next:
            next_head = head.next.next
            prev.next = head.next
            head.next.next = head
            prev = head
            head = next_head
        prev.next = head
        return dummy.next