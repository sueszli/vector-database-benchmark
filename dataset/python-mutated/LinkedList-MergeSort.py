class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.next = next

class Solution:

    def merge(self, left, right):
        if False:
            return 10
        dummy_head = ListNode(-1)
        cur = dummy_head
        while left and right:
            if left.val <= right.val:
                cur.next = left
                left = left.next
            else:
                cur.next = right
                right = right.next
            cur = cur.next
        if left:
            cur.next = left
        elif right:
            cur.next = right
        return dummy_head.next

    def mergeSort(self, head: ListNode):
        if False:
            print('Hello World!')
        if not head or not head.next:
            return head
        (slow, fast) = (head, head.next)
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
        (left_head, right_head) = (head, slow.next)
        slow.next = None
        return self.merge(self.mergeSort(left_head), self.mergeSort(right_head))

    def sortLinkedList(self, head: ListNode):
        if False:
            while True:
                i = 10
        return self.mergeSort(head)