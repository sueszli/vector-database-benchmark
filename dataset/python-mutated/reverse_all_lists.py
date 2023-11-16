"""
Reverse All Lists

Return a list that contains the items in reverse, but so that whenever each item is
itself a list, its elements are also reversed. This reversal of sublists must keep going on all the way
down, no matter how deep the nesting of these lists,

Input: [1, [2, 3, 4, 'yeah'], 5]
Output: [5, ['yeah', 4, 3, 2], 1]

=========================================
This problem can be solved using queue, stack (or recursion). Use in place reversing and save all
inner lists for reversing later.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""
from collections import deque

def reverse_all_lists(arr):
    if False:
        while True:
            i = 10
    queue = deque()
    queue.append(arr)
    while queue:
        inner_arr = queue.popleft()
        reverse_arr(inner_arr)
        for item in inner_arr:
            if isinstance(item, list):
                queue.append(item)
    return arr

def reverse_arr(arr):
    if False:
        while True:
            i = 10
    start = 0
    end = len(arr) - 1
    while start < end:
        (arr[start], arr[end]) = (arr[end], arr[start])
        start += 1
        end -= 1
    return arr
print(reverse_all_lists([1, [2, 3, 4, 'yeah'], 5]))
print(reverse_all_lists([42, [99, [17, [33, ['boo!']]]]]))