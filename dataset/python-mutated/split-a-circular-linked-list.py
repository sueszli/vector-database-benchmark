class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            i = 10
            return i + 15
        pass

class Solution(object):

    def splitCircularLinkedList(self, list):
        if False:
            for i in range(10):
                print('nop')
        '\n        :type list: Optional[ListNode]\n        :rtype: List[Optional[ListNode]]\n        '
        head1 = list
        (slow, fast) = (head1, head1.next)
        while head1 != fast.next:
            slow = slow.next
            fast = fast.next.next if head1 != fast.next.next else fast.next
        head2 = slow.next
        (slow.next, fast.next) = (head1, head2)
        return [head1, head2]