class ListNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.next = None

class Solution(object):

    def hasCycle(self, head):
        if False:
            return 10
        (fast, slow) = (head, head)
        while fast and fast.next:
            (fast, slow) = (fast.next.next, slow.next)
            if fast is slow:
                return True
        return False