class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            while True:
                i = 10
        self.val = val
        self.next = next

class Solution:

    def bubbleSort(self, head: ListNode):
        if False:
            print('Hello World!')
        node_i = head
        tail = None
        while node_i:
            node_j = head
            while node_j and node_j.next != tail:
                if node_j.val > node_j.next.val:
                    (node_j.val, node_j.next.val) = (node_j.next.val, node_j.val)
                node_j = node_j.next
            tail = node_j
            node_i = node_i.next
        return head

    def sortLinkedList(self, head: ListNode):
        if False:
            print('Hello World!')
        return self.bubbleSort(head)