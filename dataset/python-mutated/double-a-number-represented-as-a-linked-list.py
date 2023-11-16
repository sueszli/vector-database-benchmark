class Solution(object):

    def doubleIt(self, head):
        if False:
            while True:
                i = 10
        '\n        :type head: Optional[ListNode]\n        :rtype: Optional[ListNode]\n        '
        if head.val >= 5:
            head = ListNode(0, head)
        curr = head
        while curr:
            curr.val = curr.val * 2 % 10
            if curr.next and curr.next.val >= 5:
                curr.val += 1
            curr = curr.next
        return head