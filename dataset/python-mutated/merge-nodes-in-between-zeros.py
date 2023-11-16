class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            return 10
        pass

class Solution(object):

    def mergeNodes(self, head):
        if False:
            return 10
        '\n        :type head: Optional[ListNode]\n        :rtype: Optional[ListNode]\n        '
        (curr, zero) = (head.next, head)
        while curr:
            if curr.val:
                zero.val += curr.val
            else:
                zero.next = curr if curr.next else None
                zero = curr
            curr = curr.next
        return head