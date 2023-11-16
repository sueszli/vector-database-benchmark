class ListNode(object):

    def __init__(self, x):
        if False:
            while True:
                i = 10
        self.val = x
        self.next = None

class Solution(object):

    def middleNode(self, head):
        if False:
            i = 10
            return i + 15
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        (slow, fast) = (head, head)
        while fast and fast.next:
            (slow, fast) = (slow.next, fast.next.next)
        return slow