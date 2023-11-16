import math
import os
import random
import re
import sys

class SinglyLinkedListNode:

    def __init__(self, node_data):
        if False:
            return 10
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
            for i in range(10):
                print('nop')
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

def deleteNode(head, position):
    if False:
        i = 10
        return i + 15
    cur = head
    if position == 0:
        return head.next
    while position > 1:
        cur = cur.next
        position -= 1
    cur.next = cur.next.next
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    llist_count = int(input())
    llist = SinglyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)
    position = int(input())
    llist1 = deleteNode(llist.head, position)
    print_singly_linked_list(llist1, ' ', fptr)
    fptr.write('\n')
    fptr.close()