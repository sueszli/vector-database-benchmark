"""
Sort RGB Array

Given an array of strictly the characters 'R', 'G', and 'B', segregate
the values of the array so that all the Rs come first, the Gs come second, and the Bs come last.
You can only swap elements of the array.
Do this in linear time and in-place.

Input: ['G', 'B', 'R', 'R', 'B', 'R', 'G']
Output: ['R', 'R', 'R', 'G', 'G', 'B', 'B']

=========================================
Play with pointers/indices and swap elements. (only one iteration)
Save the last R, G and B indices, when adding some color, move the rest indices by 1.
    Time Complexity:    O(N)
    Space Complexity:   O(1)
Count R, G, B and populate the array after that. (2 iterations)
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def sort_rgb_array(arr):
    if False:
        print('Hello World!')
    n = len(arr)
    (r, g, b) = (0, 0, 0)
    for i in range(n):
        if arr[i] == 'R':
            swap(arr, i, r)
            r += 1
        if r > g:
            g = r
        if arr[i] == 'G':
            swap(arr, i, g)
            g += 1
        if g > b:
            b = g
        if arr[i] == 'B':
            swap(arr, i, b)
            b += 1
    return arr

def swap(arr, i, j):
    if False:
        for i in range(10):
            print('nop')
    (arr[i], arr[j]) = (arr[j], arr[i])

def sort_rgb_array_2(arr):
    if False:
        i = 10
        return i + 15
    rgb = {'R': 0, 'G': 0, 'B': 0}
    for c in arr:
        rgb[c] += 1
    rgb['G'] += rgb['R']
    rgb['B'] += rgb['G']
    for i in range(len(arr)):
        if i < rgb['R']:
            arr[i] = 'R'
        elif i < rgb['G']:
            arr[i] = 'G'
        else:
            arr[i] = 'B'
    return arr
print(sort_rgb_array(['G', 'B', 'R', 'R', 'B', 'R', 'G']))
print(sort_rgb_array_2(['G', 'B', 'R', 'R', 'B', 'R', 'G']))
print(sort_rgb_array(['B', 'B', 'B', 'G', 'G', 'R', 'R']))
print(sort_rgb_array_2(['B', 'B', 'B', 'G', 'G', 'R', 'R']))