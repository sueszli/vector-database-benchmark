class ListNode(object):

    def __init__(self, x):
        if False:
            for i in range(10):
                print('nop')
        self.val = x
        self.next = None

class Solution(object):

    def plusOne(self, head):
        if False:
            return 10
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        if not head:
            return None
        dummy = ListNode(0)
        dummy.next = head
        (left, right) = (dummy, head)
        while right.next:
            if right.val != 9:
                left = right
            right = right.next
        if right.val != 9:
            right.val += 1
        else:
            left.val += 1
            right = left.next
            while right:
                right.val = 0
                right = right.next
        return dummy if dummy.val else dummy.next

class Solution2(object):

    def plusOne(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '

        def reverseList(head):
            if False:
                return 10
            dummy = ListNode(0)
            curr = head
            while curr:
                (dummy.next, curr.next, curr) = (curr, dummy.next, curr.next)
            return dummy.next
        rev_head = reverseList(head)
        (curr, carry) = (rev_head, 1)
        while curr and carry:
            curr.val += carry
            carry = curr.val / 10
            curr.val %= 10
            if carry and curr.next is None:
                curr.next = ListNode(0)
            curr = curr.next
        return reverseList(rev_head)