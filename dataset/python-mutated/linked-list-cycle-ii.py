class ListNode(object):

    def __init__(self, x):
        if False:
            return 10
        self.val = x
        self.next = None

    def __str__(self):
        if False:
            print('Hello World!')
        if self:
            return '{}'.format(self.val)
        else:
            return None

class Solution(object):

    def detectCycle(self, head):
        if False:
            return 10
        (fast, slow) = (head, head)
        while fast and fast.next:
            (fast, slow) = (fast.next.next, slow.next)
            if fast is slow:
                fast = head
                while fast is not slow:
                    (fast, slow) = (fast.next, slow.next)
                return fast
        return None