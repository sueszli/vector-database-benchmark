"""
Remove Element

Given a linked list nums and a value val, remove all instances of that value in-place and return the new linked list.
Do not allocate extra space for another linked list, you must do this by modifying the input linked list in-place with O(1) extra memory.

Input: 3 -> 2 -> 2 -> 3
Output: 2 -> 2

Input: 0 -> 1 -> 2 -> 2 -> 3 -> 0 -> 4 -> 2
Output: 0 -> 1 -> 3 -> 0 -> 4

=========================================
Iterate the linked list and jump the values that needs to be deleted (change the next pointer).
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""
from ll_helpers import ListNode

def remove_element(nums, val):
    if False:
        for i in range(10):
            print('nop')
    res = ListNode(0)
    res.next = nums
    pointer = res
    while pointer.next is not None:
        if pointer.next.val == val:
            pointer.next = pointer.next.next
        else:
            pointer = pointer.next
    return res.next
from ll_helpers import build_ll, print_ll
print_ll(remove_element(build_ll([3, 2, 2, 3]), 3))
print_ll(remove_element(build_ll([0, 1, 2, 3, 0, 4, 2]), 2))