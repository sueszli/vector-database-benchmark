class ListNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.next = None

class Solution(object):

    def addTwoNumbers(self, l1, l2):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type l1: ListNode\n        :type l2: ListNode\n        :rtype: ListNode\n        '
        dummy = ListNode(0)
        (current, carry) = (dummy, 0)
        while l1 or l2:
            val = carry
            if l1:
                val += l1.val
                l1 = l1.next
            if l2:
                val += l2.val
                l2 = l2.next
            (carry, val) = divmod(val, 10)
            current.next = ListNode(val)
            current = current.next
        if carry == 1:
            current.next = ListNode(1)
        return dummy.next