class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.next = next

class Solution(object):

    def deleteMiddle(self, head):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type head: Optional[ListNode]\n        :rtype: Optional[ListNode]\n        '
        dummy = ListNode()
        dummy.next = head
        slow = fast = dummy
        while fast.next and fast.next.next:
            (slow, fast) = (slow.next, fast.next.next)
        slow.next = slow.next.next
        return dummy.next