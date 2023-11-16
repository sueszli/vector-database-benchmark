"""
Reverse array

Reverse an array, in constant space and linear time complexity.

Input: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
Output: [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]

=========================================
Reverse the whole array by swapping pair letters in-place (first with last, second with second from the end, etc).
Exist 2 more "Pythonic" ways of reversing arrays/strings (but not in-place, they're creating a new list):
- reversed_arr = reversed(arr)
- reversed_arr = arr[::-1]
But I wanted to show how to implement a reverse algorithm step by step so someone will know how to implement it in other languages.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def reverse_arr(arr):
    if False:
        print('Hello World!')
    start = 0
    end = len(arr) - 1
    while start < end:
        swap(arr, start, end)
        start += 1
        end -= 1
    return arr

def swap(arr, i, j):
    if False:
        i = 10
        return i + 15
    (arr[i], arr[j]) = (arr[j], arr[i])
    'same as\n    temp = arr[i]\n    arr[i] = arr[j]\n    arr[j] = temp\n    '
print(reverse_arr([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]))
print(reverse_arr([1, 2, 3, 4, 5]))