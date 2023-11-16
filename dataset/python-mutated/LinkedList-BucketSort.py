class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            for i in range(10):
                print('nop')
        self.val = val
        self.next = next

class Solution:

    def insertion(self, buckets, index, val):
        if False:
            while True:
                i = 10
        if not buckets[index]:
            buckets[index] = ListNode(val)
            return
        node = ListNode(val)
        node.next = buckets[index]
        buckets[index] = node

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

    def bucketSort(self, head: ListNode, bucket_size=5):
        if False:
            while True:
                i = 10
        if not head:
            return head
        (list_min, list_max) = (float('inf'), float('-inf'))
        cur = head
        while cur:
            if cur.val < list_min:
                list_min = cur.val
            if cur.val > list_max:
                list_max = cur.val
            cur = cur.next
        bucket_count = (list_max - list_min) // bucket_size + 1
        buckets = [None for _ in range(bucket_count)]
        cur = head
        while cur:
            index = (cur.val - list_min) // bucket_size
            self.insertion(buckets, index, cur.val)
            cur = cur.next
        dummy_head = ListNode(-1)
        cur = dummy_head
        for bucket_head in buckets:
            bucket_cur = self.mergeSort(bucket_head)
            while bucket_cur:
                cur.next = bucket_cur
                cur = cur.next
                bucket_cur = bucket_cur.next
        return dummy_head.next

    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if False:
            while True:
                i = 10
        return self.bucketSort(head)