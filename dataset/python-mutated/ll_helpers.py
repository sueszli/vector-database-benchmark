"""
Helpers used in all linked list solutions.
This file is created to avoid duplicate code in all linked list solutions.
"""

class ListNode:

    def __init__(self, x, n=None):
        if False:
            while True:
                i = 10
        'Definition for singly-linked list.'
        self.val = x
        self.next = n

def build_ll(arr):
    if False:
        for i in range(10):
            print('nop')
    'Builds a linked list from array. Used for testing.'
    res = ListNode(None)
    pt = res
    for num in arr:
        pt.next = ListNode(num)
        pt = pt.next
    return res.next

def print_ll(head):
    if False:
        return 10
    'Prints a linked list in this format: x -> y -> z. Used for testing.'
    res = []
    while head != None:
        res.append(str(head.val))
        head = head.next
    print(' -> '.join(res))