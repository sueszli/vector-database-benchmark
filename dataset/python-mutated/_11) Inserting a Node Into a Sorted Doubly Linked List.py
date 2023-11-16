import math
import os
import random
import re
import sys

class DoublyLinkedListNode:

    def __init__(self, node_data):
        if False:
            i = 10
            return i + 15
        self.data = node_data
        self.next = None
        self.prev = None

class DoublyLinkedList:

    def __init__(self):
        if False:
            i = 10
            return i + 15
        self.head = None
        self.tail = None

    def insert_node(self, node_data):
        if False:
            return 10
        node = DoublyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
            node.prev = self.tail
        self.tail = node

def print_doubly_linked_list(node, sep, fptr):
    if False:
        for i in range(10):
            print('nop')
    while node:
        fptr.write(str(node.data))
        node = node.next
        if node:
            fptr.write(sep)

def sortedInsert(head, data):
    if False:
        i = 10
        return i + 15
    newNode = DoublyLinkedListNode(data)
    if not head:
        head = newNode
        return head
    cur = head
    if cur.data >= data:
        newNode.next = cur
        cur.prev = newNode
        head = newNode
        return head
    else:
        while cur.next:
            if cur.data < data and cur.next.data >= data:
                newNode.prev = cur
                newNode.next = cur.next
                cur.next = newNode
                cur.next.prev = newNode
                return head
            else:
                cur = cur.next
        else:
            cur.next = newNode
            newNode.prev = cur
    return head
if __name__ == '__main__':
    fptr = open(os.environ['OUTPUT_PATH'], 'w')
    t = int(input())
    for t_itr in range(t):
        llist_count = int(input())
        llist = DoublyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        data = int(input())
        llist1 = sortedInsert(llist.head, data)
        print_doubly_linked_list(llist1, ' ', fptr)
        fptr.write('\n')
    fptr.close()