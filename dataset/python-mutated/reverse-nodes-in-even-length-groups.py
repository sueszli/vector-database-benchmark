class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.next = next

class Solution(object):

    def reverseEvenLengthGroups(self, head):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type head: Optional[ListNode]\n        :rtype: Optional[ListNode]\n        '
        (prev, l) = (head, 2)
        while prev.next:
            (curr, cnt) = (prev, 0)
            for _ in xrange(l):
                if not curr.next:
                    break
                cnt += 1
                curr = curr.next
            l += 1
            if cnt % 2:
                prev = curr
                continue
            (curr, last) = (prev.next, None)
            for _ in xrange(cnt):
                (curr.next, curr, last) = (last, curr.next, curr)
            (prev.next.next, prev.next, prev) = (curr, last, prev.next)
        return head