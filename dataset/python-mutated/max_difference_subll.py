"""
Maximum Difference Sub-Linked List

Given a linked list of integers, find and return the sub-linked list of k consecutive elements where
the difference between the smallest element and the largest element is the largest possible.
If there are several sub-linked lists of k elements in items so that all these sub-linked list have
the same largest possible difference, return the sub-linked list that occurs first.

Input: 42 -> 17 -> 99 -> 12 -> 65 -> 77 -> 11 -> 26, 5
Output: 99 -> 12 -> 65 -> 77 -> 11

=========================================
Using 2 pointers (start and end), traverse the linked list and compare the results.
But first, move the end pointer for k places.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""
from ll_helpers import ListNode

def max_diference_subll(ll, k):
    if False:
        return 10
    if ll is None:
        return None
    (start, end) = (ll, ll)
    for i in range(1, k):
        end = end.next
        if end is None:
            return None
    (result_start, result_end) = (start, end)
    while end is not None:
        if abs(result_start.val - result_end.val) < abs(start.val - end.val):
            (result_start, result_end) = (start, end)
        start = start.next
        end = end.next
    result_end.next = None
    return result_start
from ll_helpers import build_ll, print_ll
print_ll(max_diference_subll(build_ll([42, 17, 99, 12, 65, 77, 11, 26]), 5))
print_ll(max_diference_subll(build_ll([36, 14, 58, 11, 63, 77, 46, 32, 87]), 5))