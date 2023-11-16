"""
Find the missing number in a sequence

Find the only missing integer in a sequence,
all numbers are integers and they're smaller or equal to N+1 (N is length of the array).

Input: [2, 1, 4]
Output: 3

=========================================
Searching for 1 unknown, math problem.
Use the sum formula for the first N numbers to compute the whole sum of the sequence.
After that sum all elements from the array, and when you subtract those 2 numbers, you'll get the missing number.
Sum formula = N*(N+1)/2
    Time Complexity:    O(N)
    Space Complexity:   O(1)
"""

def missing_number(nums):
    if False:
        while True:
            i = 10
    s = sum(nums)
    n = len(nums) + 1
    return n * (n + 1) // 2 - s
print(missing_number([2, 3, 1]))
print(missing_number([2, 1, 4]))