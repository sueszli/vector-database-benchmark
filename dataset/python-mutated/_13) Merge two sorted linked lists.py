import math
import os
import random
import re
import sys

class SinglyLinkedListNode:

    def __init__(self, node_data):
        if False:
            print('Hello World!')
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
            while True:
                i = 10
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node

def print_singly_linked_list(node, sep, fptr):
    if False:
        i = 10
        return i + 15
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)

def mergeLists(head1, head2):
    if False:
        while True:
            i = 10
    if not head1:
        return head2
    if not head2:
        return head1
    if head1.data <= head2.data:
        head = head1
        cur1 = head1.next
        cur2 = head2
    else:
        head = head2
        cur1 = head1
        cur2 = head2.next
    cur = head
    while True:
        if cur1 is None:
            cur.next = cur2
            break
        elif cur2 is None:
            cur.next = cur1
            break
        if cur1.data <= cur2.data:
            cur.next = cur1
            cur1 = cur1.next
        else:
            cur.next = cur2
            cur2 = cur2.next
        cur = cur.next
    return head
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
        llist3 = mergeLists(llist1.head, llist2.head)
        print_singly_linked_list(llist3, ' ', fptr)
        fptr.write('\n')
    fptr.close()