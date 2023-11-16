class ListNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.next = None

    def __repr__(self):
        if False:
            i = 10
            return i + 15
        if self:
            return '{} -> {}'.format(self.val, self.next)

class Solution(object):

    def mergeTwoLists(self, l1, l2):
        if False:
            print('Hello World!')
        '\n        :type l1: ListNode\n        :type l2: ListNode\n        :rtype: ListNode\n        '
        curr = dummy = ListNode(0)
        while l1 and l2:
            if l1.val < l2.val:
                curr.next = l1
                l1 = l1.next
            else:
                curr.next = l2
                l2 = l2.next
            curr = curr.next
        curr.next = l1 or l2
        return dummy.next