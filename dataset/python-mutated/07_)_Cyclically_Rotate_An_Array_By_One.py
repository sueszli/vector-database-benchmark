"""
INPUT : [1, 2, 3, 4, 5]
OUTPUT : [5, 1, 2, 3, 4]
Just rotate the array by one
"""

def rotate(arr):
    if False:
        return 10
    '\n    Time Complexity : O(n)\n    Space Complexity : O(1)\n    '
    return arr.insert(0, arr.pop())
arr = [1, 2, 3, 4, 5]
rotate(arr)
print(arr)