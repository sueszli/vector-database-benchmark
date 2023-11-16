import math
import os
import random
import re
import sys

class SinglyLinkedListNode:

    def __init__(self, node_data):
        if False:
            i = 10
            return i + 15
        self.data = node_data
        self.next = None

class SinglyLinkedList:

    def __init__(self):
        if False:
            return 10
        self.head = None
        self.tail = None

    def insert_node(self, node_data):
        if False:
            print('Hello World!')
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node

def print_singly_linked_list(node, sep, fptr):
    if False:
        while True:
            i = 10
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)

def removeDuplicates(head):
    if False:
        for i in range(10):
            print('nop')
    cur = head
    while cur.next:
        if cur.data == cur.next.data:
            cur.next = cur.next.next
        else:
            cur = cur.next
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        llist1 = removeDuplicates(llist.head)
        print_singly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()