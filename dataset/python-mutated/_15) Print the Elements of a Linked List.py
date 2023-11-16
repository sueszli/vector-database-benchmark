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
            i = 10
            return i + 15
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

def printLinkedList(head):
    if False:
        print('Hello World!')
    if not head:
        return
    while head != None:
        print(head.data)
        head = head.next
if __name__ == '__main__':
    llist_count = int(input())
    llist = SinglyLinkedList()
    for _ in range(llist_count):
        llist_item = int(input())
        llist.insert_node(llist_item)
    printLinkedList(llist.head)