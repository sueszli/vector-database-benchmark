"""
Reverse a linked list

Reverse a linked list in one iteration without using additional space.

Input: 1 -> 2 -> 3 -> 4
Output: 4 -> 3 -> 2 -> 1

=========================================
Iterate LL and change the pointer of the current nodes to point to the previous nodes.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
Solution 2: Same approach using recursion.
    Time Complexity:    O(N)
    Space Complexity:   O(N)        , because of the recursion stack (the stack will be with N depth till the last node of the linked list is reached)
"""
from ll_helpers import ListNode

def reverse_ll(ll):
    if False:
        i = 10
        return i + 15
    prev_node = None
    while ll is not None:
        current = ll
        ll = ll.next
        current.next = prev_node
        prev_node = current
    return prev_node

def reverse_ll_2(ll):
    if False:
        print('Hello World!')
    return reverse(ll, None)

def reverse(node, prev_node):
    if False:
        for i in range(10):
            print('nop')
    if node is None:
        return prev_node
    result = reverse(node.next, node)
    node.next = prev_node
    return result
from ll_helpers import build_ll, print_ll
print_ll(reverse_ll(build_ll([1, 2, 3, 4])))
print_ll(reverse_ll_2(build_ll([1, 2, 3, 4])))