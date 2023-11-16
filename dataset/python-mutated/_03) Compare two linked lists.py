import os
import sys

class SinglyLinkedListNode:

    def __init__(self, node_data):
        if False:
            for i in range(10):
                print('nop')
        self.data = node_data
        self.next = None

class SinglyLinkedList:

    def __init__(self):
        if False:
            print('Hello World!')
        self.head = None
        self.tail = None

    def insert_node(self, node_data):
        if False:
            i = 10
            return i + 15
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node

def print_singly_linked_list(node, sep, fptr):
    if False:
        return 10
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)

def compare_lists(head1, head2):
    if False:
        for i in range(10):
            print('nop')
    if not head1 or not head2:
        return '0'
    while head1 and head2:
        if head1.data != head2.data:
            return '0'
        head1 = head1.next
        head2 = head2.next
    if head1 or head2:
        return '0'
    return '1'
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        llist1_count = int(input())
        llist1 = SinglyLinkedList()
        for _ in range(llist1_count):
            llist1_item = int(input())
            llist1.insert_node(llist1_item)
        llist2_count = int(input())
        llist2 = SinglyLinkedList()
        for _ in range(llist2_count):
            llist2_item = int(input())
            llist2.insert_node(llist2_item)
        result = compare_lists(llist1.head, llist2.head)
        fptr.write(str(int(result)) + '\n')
    fptr.close()