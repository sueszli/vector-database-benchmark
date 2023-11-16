"""
Add Two Numbers

You are given two non-empty linked lists representing two non-negative integers.
The digits are stored in reverse order and each of their nodes contain a single digit.
Add the two numbers and return it as a linked list.
You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Input: 2 -> 4 -> 3, 5 -> 6 -> 4
Output: 7 -> 0 -> 8
Output explanation: 342 + 465 = 807

=========================================
Iterate LL and add values on same position (just like adding real numbers).
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""
from ll_helpers import ListNode

def add_two_numbers(l1, l2):
    if False:
        i = 10
        return i + 15
    start = ListNode(None)
    start.next = l1
    pointer = start
    transfer = 0
    while l1 is not None or l2 is not None or transfer != 0:
        v1 = 0
        if l1 is not None:
            v1 = l1.val
            l1 = l1.next
        v2 = 0
        if l2 is not None:
            v2 = l2.val
            l2 = l2.next
        total = transfer + v1 + v2
        transfer = total // 10
        if l1 is None:
            pointer.next = ListNode(None)
        pointer = pointer.next
        pointer.val = total % 10
    return start.next
from ll_helpers import build_ll, print_ll
ll1 = build_ll([2, 4, 3])
ll2 = build_ll([5, 6, 4])
print_ll(add_two_numbers(ll1, ll2))
ll1 = build_ll([9, 9, 9, 9])
ll2 = build_ll([9, 9])
print_ll(add_two_numbers(ll1, ll2))