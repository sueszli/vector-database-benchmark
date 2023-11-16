class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.next = next

class Solution:

    def radixSort(self, head: ListNode):
        if False:
            for i in range(10):
                print('nop')
        size = 0
        cur = head
        while cur:
            val_len = len(str(cur.val))
            if val_len > size:
                size = val_len
            cur = cur.next
        for i in range(size):
            buckets = [[] for _ in range(10)]
            cur = head
            while cur:
                buckets[cur.val // 10 ** i % 10].append(cur.val)
                cur = cur.next
            dummy_head = ListNode(-1)
            cur = dummy_head
            for bucket in buckets:
                for num in bucket:
                    cur.next = ListNode(num)
                    cur = cur.next
            head = dummy_head.next
        return head

    def sortLinkedList(self, head: ListNode):
        if False:
            while True:
                i = 10
        return self.radixSort(head)