"""
ZigZag Conversion

The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)

P   A   H   N
A P L S I I G
Y   I   R
And then read line by line: 'PAHNAPLSIIGYIR'

Input: s = 'PAYPALISHIRING', num_rows = 3
Output: 'PAHNAPLSIIGYIR'

=========================================
Go row by row and using the steps logic build the new string by jumping chars.
Middle rows have 2 times more elements than the first and last row.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
Collect all parts in separate bucket, in the end merge these buckets.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def convert(s, num_rows):
    if False:
        return 10
    if num_rows == 1:
        return s
    n = len(s)
    res = ''
    cycle = 2 * (num_rows - 1)
    for i in range(0, num_rows):
        steps = cycle - 2 * i
        if i == 0 or i == num_rows - 1:
            steps = cycle
        j = i
        while j < n:
            res += s[j]
            j += steps
            if i > 0 and i < num_rows - 1:
                steps = cycle - steps
    return res

def convert_2(word, numRows):
    if False:
        print('Hello World!')
    numLetters = len(word)
    bucket = [''] * numRows
    fullCycle = 2 * (numRows - 1)
    if numRows == 1:
        fullCycle = 1
    for pos in range(0, numLetters):
        posCycle = pos % fullCycle
        if posCycle >= numRows:
            posCycle = fullCycle - posCycle
        bucket[posCycle] += word[pos]
    result = ''
    for part in bucket:
        result += part
    return result
print(convert('PAYPALISHIRING', 3))
print(convert_2('PAYPALISHIRING', 3))
print(convert('PAYPALISHIRING', 4))
print(convert_2('PAYPALISHIRING', 4))