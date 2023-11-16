"""
Reverse string

Reverse string, in linear time complexity.

Input: 'i like this program very much'
Output: 'hcum yrev margorp siht ekil i'

Input: 'how are you'
Output: 'uoy era woh'

=========================================
Reverse the whole sentence by swapping pair letters in-place (first with last, second with second from the end, etc).
In Python, the string manipulation operations are too slow (string is immutable), because of that we need to convert the string into array.
In C/C++, the Space complexity will be O(1) (because the strings are just arrays with chars).
Exist 2 more "Pythonic" ways of reversing strings/arrays:
- reversed_str = reversed(str)
- reversed_str = str[::-1]
But I wanted to show how to implement a reverse algorithm step by step so someone will know how to implement it in other languages.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def reverse_sentence(sentence):
    if False:
        return 10
    arr = [c for c in sentence]
    start = 0
    end = len(arr) - 1
    while start < end:
        swap(arr, start, end)
        start += 1
        end -= 1
    return ''.join(arr)

def swap(arr, i, j):
    if False:
        return 10
    (arr[i], arr[j]) = (arr[j], arr[i])
    'same as\n    temp = arr[i]\n    arr[i] = arr[j]\n    arr[j] = temp\n    '
print(reverse_sentence('i like this program very much'))
print(reverse_sentence('how are you'))