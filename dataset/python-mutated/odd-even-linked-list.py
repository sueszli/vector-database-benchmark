class Solution(object):

    def oddEvenList(self, head):
        if False:
            i = 10
            return i + 15
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        if head:
            (odd_tail, cur) = (head, head.next)
            while cur and cur.next:
                even_head = odd_tail.next
                odd_tail.next = cur.next
                odd_tail = odd_tail.next
                cur.next = odd_tail.next
                odd_tail.next = even_head
                cur = cur.next
        return head