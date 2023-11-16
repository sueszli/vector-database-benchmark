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
            while True:
                i = 10
        node = SinglyLinkedListNode(node_data)
        if not self.head:
            self.head = node
        else:
            self.tail.next = node
        self.tail = node

def print_singly_linked_list(node, sep):
    if False:
        print('Hello World!')
    while node:
        print(node.data, end='')
        node = node.next
        if node:
            print(sep, end='')

def reversePrint(head):
    if False:
        print('Hello World!')
    if head:
        reversePrint(head.next)
        print(head.data)
    else:
        return
if __name__ == '__main__':
    tests = int(input())
    for tests_itr in range(tests):
        llist_count = int(input())
        llist = SinglyLinkedList()
        for _ in range(llist_count):
            llist_item = int(input())
            llist.insert_node(llist_item)
        reversePrint(llist.head)