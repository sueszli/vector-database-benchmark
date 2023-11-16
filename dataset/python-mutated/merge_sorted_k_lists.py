"""
Merge k sorted linked lists and return it as one sorted list. Analyze and describe its complexity.
"""
from heapq import heappop, heapreplace, heapify
from queue import PriorityQueue

class ListNode(object):
    """ ListNode Class"""

    def __init__(self, val):
        if False:
            i = 10
            return i + 15
        self.val = val
        self.next = None

def merge_k_lists(lists):
    if False:
        for i in range(10):
            print('nop')
    ' Merge Lists '
    dummy = node = ListNode(0)
    list_h = [(n.val, n) for n in lists if n]
    heapify(list_h)
    while list_h:
        (_, n_val) = list_h[0]
        if n_val.next is None:
            heappop(list_h)
        else:
            heapreplace(list_h, (n_val.next.val, n_val.next))
        node.next = n_val
        node = node.next
    return dummy.next

def merge_k_lists(lists):
    if False:
        print('Hello World!')
    ' Merge List '
    dummy = ListNode(None)
    curr = dummy
    q = PriorityQueue()
    for node in lists:
        if node:
            q.put((node.val, node))
    while not q.empty():
        curr.next = q.get()[1]
        curr = curr.next
        if curr.next:
            q.put((curr.next.val, curr.next))
    return dummy.next
"\nI think my code's complexity is also O(nlogk) and not using heap or priority queue,\nn means the total elements and k means the size of list.\n\nThe mergeTwoLists function in my code comes from the problem Merge Two Sorted Lists\nwhose complexity obviously is O(n), n is the sum of length of l1 and l2.\n\nTo put it simpler, assume the k is 2^x, So the progress of combination is like a full binary tree,\nfrom bottom to top. So on every level of tree, the combination complexity is n,\nbecause every level have all n numbers without repetition.\nThe level of tree is x, ie log k. So the complexity is O(n log k).\n\nfor example, 8 ListNode, and the length of every ListNode is x1, x2,\nx3, x4, x5, x6, x7, x8, total is n.\n\non level 3: x1+x2, x3+x4, x5+x6, x7+x8 sum: n\n\non level 2: x1+x2+x3+x4, x5+x6+x7+x8 sum: n\n\non level 1: x1+x2+x3+x4+x5+x6+x7+x8 sum: n\n"