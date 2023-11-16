class ListNode(object):

    def __init__(self, x):
        if False:
            i = 10
            return i + 15
        self.val = x
        self.next = None

class Solution(object):

    def deleteDuplicates(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        cur = head
        while cur:
            runner = cur.next
            while runner and runner.val == cur.val:
                runner = runner.next
            cur.next = runner
            cur = runner
        return head

    def deleteDuplicates2(self, head):
        if False:
            return 10
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        if not head:
            return head
        if head.next:
            if head.val == head.next.val:
                head = self.deleteDuplicates2(head.next)
            else:
                head.next = self.deleteDuplicates2(head.next)
        return head