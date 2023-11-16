"""
Reverse Every Ascending Sublist

Create and return a new list that contains the same elements as the argument list items, but
reversing the order of the elements inside every maximal strictly ascending sublist

Input: [5, 7, 10, 4, 2, 7, 8, 1, 3]
Output: [10, 7, 5, 4, 8, 7, 2, 3, 1]
Output explanation: 5, 7, 10 => 10, 7, 5 ; 4 => 4; 2, 7, 8 => 8, 7, 2; 1, 3 => 3, 1

=========================================
Find the start and end of each sublist and reverse it in-place.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def reverse_ascending_sublists(arr):
    if False:
        while True:
            i = 10
    n = len(arr)
    if n == 0:
        return []
    start = 0
    for i in range(1, n):
        if arr[i] < arr[i - 1]:
            reverse_arr(arr, start, i - 1)
            start = i
    reverse_arr(arr, start, n - 1)
    return arr

def reverse_arr(arr, start, end):
    if False:
        return 10
    while start < end:
        (arr[start], arr[end]) = (arr[end], arr[start])
        start += 1
        end -= 1
    return arr
print(reverse_ascending_sublists([1, 2, 3, 4, 5]))
print(reverse_ascending_sublists([5, 4, 3, 2, 1]))
print(reverse_ascending_sublists([5, 7, 10, 4, 2, 7, 8, 1, 3]))