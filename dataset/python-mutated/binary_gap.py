"""
Given a positive integer N, find and return the longest distance between two
consecutive 1' in the binary representation of N.
If there are not two consecutive 1's, return 0

For example:
Input: 22
Output: 2
Explanation:
22 in binary is 10110
In the binary representation of 22, there are three ones, and two consecutive pairs of 1's.
The first consecutive pair of 1's have distance 2.
The second consecutive pair of 1's have distance 1.
The answer is the largest of these two distances, which is 2
"""

def binary_gap(N):
    if False:
        print('Hello World!')
    last = None
    ans = 0
    index = 0
    while N != 0:
        if N & 1:
            if last is not None:
                ans = max(ans, index - last)
            last = index
        index = index + 1
        N = N >> 1
    return ans

def binary_gap_improved(N):
    if False:
        print('Hello World!')
    last = None
    ans = 0
    index = 0
    while N != 0:
        tes = N & 1
        if tes:
            if last is not None:
                ans = max(ans, index - last + 1)
            else:
                last = index
        else:
            last = index + 1
        index = index + 1
        N = N >> 1
    return ans
print(binary_gap(111))
print(binary_gap_improved(111))