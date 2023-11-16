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
            for i in range(10):
                print('nop')
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

def reverse(head):
    if False:
        while True:
            i = 10
    prevPointer = None
    currentPointer = head
    while currentPointer is not None:
        nextPointer = currentPointer.next
        currentPointer.next = prevPointer
        prevPointer = currentPointer
        currentPointer = nextPointer
    head = prevPointer
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    tests = int(input())
    for tests_itr in range(tests):
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        llist1 = reverse(llist.head)
        print_singly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()