"""
Remove Nth Node From End of List

Given a linked list, remove the n-th node from the end of list and return its head.

Input: 1 -> 2 -> 3 -> 4 -> 5, 2.
Output: 1 -> 2 -> 3 -> 5

=========================================
Playing with the pointers.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
Recursive solution.
    Time Complexity:    O(N)
    Space Complexity:   O(N)  , because of the recursive calls stack
"""
from ll_helpers import ListNode

def remove_nth_from_end_1(head, n):
    if False:
        i = 10
        return i + 15
    helper = ListNode(0)
    helper.next = head
    first = helper
    second = helper
    for i in range(n + 1):
        first = first.next
    while first != None:
        first = first.next
        second = second.next
    second.next = second.next.next
    return helper.next

def remove_nth_from_end_2(head, n):
    if False:
        i = 10
        return i + 15
    result = remove_recursively(head, n)
    if result[0] == n:
        return head.next
    return head

def remove_recursively(pointer, n):
    if False:
        print('Hello World!')
    if pointer is None:
        return (0, None)
    result = remove_recursively(pointer.next, n)
    if result[0] == n:
        pointer.next = result[1]
    return (result[0] + 1, pointer.next)