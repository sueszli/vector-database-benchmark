"""
EXAMPLE :
INPUT : [0,2,2,1,0]
OUTPUT : [0,0,1,2,2]
"""

def sort012(arr):
    if False:
        return 10
    '\n    Time Complexity : O(n) (But only one traversals is required)\n    Space Complexity : O(1)\n    '
    (low, mid, high) = (0, 0, len(arr) - 1)
    while mid <= high:
        if arr[mid] == 0:
            (arr[mid], arr[low]) = (arr[low], arr[mid])
            low += 1
            mid += 1
        elif arr[mid] == 1:
            mid += 1
        else:
            (arr[mid], arr[high]) = (arr[high], arr[mid])
            high -= 1
    return arr
print(sort012([0, 2, 1, 2, 0]))