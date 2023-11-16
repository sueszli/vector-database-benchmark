"""
Intersecting Linked Lists

Given two singly linked lists that intersect at some point, find the intersecting node. The lists are non-cyclical.
In this example, assume nodes with the same value are the exact same node objects.

Input: 3 -> 7 -> 8 -> 10, 99 -> 1 -> 8 -> 10
Output: 8

=========================================
Find the longer linked list and move the pointer (now both list will have same number of elements).
After that move both pointers from the both lists and compare elements.
    Time Complexity:    O(N + M)
    Space Complexity:   O(1)
"""
from ll_helpers import ListNode

def find_intersecting_node(ll1, ll2):
    if False:
        for i in range(10):
            print('nop')
    count1 = 0
    temp1 = ll1
    while temp1 is not None:
        count1 += 1
        temp1 = temp1.next
    count2 = 0
    temp2 = ll2
    while temp2 is not None:
        count2 += 1
        temp2 = temp2.next
    m = min(count1, count2)
    for i in range(count1 - m):
        ll1 = ll1.next
    for i in range(count2 - m):
        ll2 = ll2.next
    intersect = None
    while ll1 is not None:
        if ll1.val != ll2.val:
            intersect = None
        elif intersect == None:
            intersect = ll1
        ll1 = ll1.next
        ll2 = ll2.next
    return intersect
from ll_helpers import build_ll
ll1 = build_ll([3, 7, 8, 10])
ll2 = build_ll([1, 8, 10])
print(find_intersecting_node(ll1, ll2).val)