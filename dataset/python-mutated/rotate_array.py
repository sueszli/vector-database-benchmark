"""
Array rotation/shifting

Rotate array in right (or left) for K places.

Input: [1, 2, 3, 4, 5, 6], 1
Output: [6, 1, 2, 3, 4, 5]

Input: [1, 2, 3, 4, 5, 6], 3
Output: [4, 5, 6, 1, 2, 3]

=========================================
The first solution is a simple one, split the array in two parts and swap those parts.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
For the second one we need to compute GCD, to decide how many different sets are there.
And after that shift all elements in that set for one position in right/left.
(elements in a set are not neighboring elements)
(A Juggling Algorithm, https://www.geeksforgeeks.org/array-rotation/)
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def rotate_array_1(arr, k, right=True):
    if False:
        print('Hello World!')
    n = len(arr)
    right %= n
    if right:
        k = n - k
    return arr[k:] + arr[:k]

def rotate_array_2(arr, k, right=True):
    if False:
        i = 10
        return i + 15
    n = len(arr)
    right %= n
    if not right:
        k = n - k
    sets = gcd(n, k)
    elements = n // sets
    i = 0
    while i < sets:
        j = 1
        curr = arr[i]
        while j <= elements:
            idx = (i + j * k) % n
            j += 1
            (curr, arr[idx]) = (arr[idx], curr)
            'same as\n            temp = curr\n            curr = arr[idx]\n            arr[idx] = temp\n            '
        i += 1
    return arr

def gcd(a, b):
    if False:
        return 10
    if b == 0:
        return a
    return gcd(b, a % b)
arr = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
k = 7
print(rotate_array_1(arr, k))
print(rotate_array_2(arr, k))
arr = [1, 2, 3, 4, 5, 6]
k = 1
print(rotate_array_1(arr, k))
print(rotate_array_2(arr, k))
arr = [1, 2, 3, 4, 5, 6]
k = 3
print(rotate_array_1(arr, k))
print(rotate_array_2(arr, k))