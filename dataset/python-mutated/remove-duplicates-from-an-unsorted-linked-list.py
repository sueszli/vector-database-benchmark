import collections

class ListNode(object):

    def __init__(self, val=0, next=None):
        if False:
            return 10
        self.val = val
        self.next = next

class Solution(object):

    def deleteDuplicatesUnsorted(self, head):
        if False:
            print('Hello World!')
        '\n        :type head: ListNode\n        :rtype: ListNode\n        '
        count = collections.defaultdict(int)
        curr = head
        while curr:
            count[curr.val] += 1
            curr = curr.next
        curr = dummy = ListNode(0, head)
        while curr.next:
            if count[curr.next.val] == 1:
                curr = curr.next
            else:
                curr.next = curr.next.next
        return dummy.next