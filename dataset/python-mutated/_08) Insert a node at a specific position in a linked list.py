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
        while True:
            i = 10
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)

def insertNodeAtPosition(head, data, position):
    if False:
        i = 10
        return i + 15
    cur = head
    newNode = SinglyLinkedListNode(data)
    if position == 0:
        newNode.next = cur
        head = newNode
        return head
    while position > 1:
        position -= 1
        cur = cur.next
    newNode.next = cur.next
    cur.next = newNode
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)
    data = int(input())
    position = int(input())
    llist_head = insertNodeAtPosition(llist.head, data, position)
    print_singly_linked_list(llist_head, ' ', fptr)
    fptr.write('\n')
    fptr.close()