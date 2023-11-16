class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            return 10
        if self:
            return '{} -> {}'.format(self.val, self.next)

class Solution(object):

    def swapPairs(self, head):
        if False:
            return 10
        dummy = ListNode(0)
        dummy.next = head
        current = dummy
        while current.next and current.next.next:
            (next_one, next_two, next_three) = (current.next, current.next.next, current.next.next.next)
            current.next = next_two
            next_two.next = next_one
            next_one.next = next_three
            current = next_one
        return dummy.next