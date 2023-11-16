"""
Remove Duplicates from Sorted Linked List

Given a sorted linked list nums, remove the duplicates in-place such that each element appear only once and return the modified linked list.
Do not allocate extra space for another linked list, you must do this by modifying the input linked list in-place with O(1) extra memory.

Input: 1 -> 1 -> 2
Output: 1 -> 2

Input: 0 -> 0 -> 1 -> 1 -> 1 -> 2 -> 2 -> 3 -> 3 -> 4
Output: 0 -> 1 -> 2 -> 3 -> 4

=========================================
Iterate the linked list and jump the neighbouring duplicates (change the next pointer).
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""
from ll_helpers import ListNode

def remove_duplicates(nums):
    if False:
        i = 10
        return i + 15
    if nums is None:
        return nums
    pointer = nums
    while pointer.next is not None:
        if pointer.val == pointer.next.val:
            pointer.next = pointer.next.next
        else:
            pointer = pointer.next
    return nums
from ll_helpers import build_ll, print_ll
print_ll(remove_duplicates(build_ll([1, 1, 2])))
print_ll(remove_duplicates(build_ll([0, 0, 1, 1, 1, 2, 2, 3, 3, 4])))