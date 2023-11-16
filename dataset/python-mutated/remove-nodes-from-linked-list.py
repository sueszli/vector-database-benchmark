class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def removeNodes(self, head):
        if False:
            return 10
        '\n        :type head: Optional[ListNode]\n        :rtype: Optional[ListNode]\n        '
        stk = []
        while head:
            while stk and stk[-1].val < head.val:
                stk.pop()
            if stk:
                stk[-1].next = head
            stk.append(head)
            head = head.next
        return stk[0]