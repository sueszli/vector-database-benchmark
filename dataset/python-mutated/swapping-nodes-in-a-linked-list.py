class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            print('Hello World!')
        pass

class Solution(object):

    def swapNodes(self, head, k):
        if False:
            return 10
        '\n        :type head: ListNode\n        :type k: int\n        :rtype: ListNode\n        '
        (left, right, curr) = (None, None, head)
        while curr:
            k -= 1
            if right:
                right = right.next
            if k == 0:
                left = curr
                right = head
            curr = curr.next
        (left.val, right.val) = (right.val, left.val)
        return head