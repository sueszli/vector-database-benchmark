class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.next = next

class Solution:

    def mergeTwoLists(self, l1: ListNode, l2: ListNode) -> ListNode:
        if False:
            return 10
        prev = dummy = ListNode(None)
        while l1 and l2:
            if l1.val < l2.val:
                prev.next = l1
                l1 = l1.next
            else:
                prev.next = l2
                l2 = l2.next
            prev = prev.next
        prev.next = l1 or l2
        return dummy.next