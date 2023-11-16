class ListNode:

    def __init__(self, val=0, next=None):
        if False:
            return 10
        self.val = val
        self.next = next

class Solution:

    def sectionSort(self, head: ListNode):
        if False:
            for i in range(10):
                print('nop')
        node_i = head
        while node_i and node_i.next:
            min_node = node_i
            node_j = node_i.next
            while node_j:
                if node_j.val < min_node.val:
                    min_node = node_j
                node_j = node_j.next
            if node_i != min_node:
                (node_i.val, min_node.val) = (min_node.val, node_i.val)
            node_i = node_i.next
        return head

    def sortLinkedList(self, head: ListNode):
        if False:
            i = 10
            return i + 15
        return self.sectionSort(head)