class ListNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.next = None

class Solution(object):

    def addTwoNumbers(self, l1, l2):
        if False:
            print('Hello World!')
        '\n        :type l1: ListNode\n        :type l2: ListNode\n        :rtype: ListNode\n        '
        (stk1, stk2) = ([], [])
        while l1:
            stk1.append(l1.val)
            l1 = l1.next
        while l2:
            stk2.append(l2.val)
            l2 = l2.next
        (prev, head) = (None, None)
        sum = 0
        while stk1 or stk2:
            sum /= 10
            if stk1:
                sum += stk1.pop()
            if stk2:
                sum += stk2.pop()
            head = ListNode(sum % 10)
            head.next = prev
            prev = head
        if sum >= 10:
            head = ListNode(sum / 10)
            head.next = prev
        return head