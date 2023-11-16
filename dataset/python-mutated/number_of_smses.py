"""
Number of SMSes

Given the number sequence that is being typed in order to write and SMS message, return the count
of all the possible messages that can be constructed.

 1    2    3
     abc  def

 4    5    6
ghi  jkl  mno

 7    8    9
pqrs tuv wxyz

The blank space character is constructed with a '0'.

Input: '222'
Output: 4
Output explanation: '222' could mean: 'c', 'ab','ba' or 'aaa'. That makes 4 possible messages.

=========================================
Dynamic programming solution. Similar to number_of_decodings.py.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def num_smses(sequence):
    if False:
        for i in range(10):
            print('nop')
    n = len(sequence)
    dp = [0] * n
    for i in range(min(4, n)):
        if is_valid(sequence[0:i + 1]):
            dp[i] = 1
    for i in range(1, n):
        for j in range(min(4, i)):
            if is_valid(sequence[i - j:i + 1]):
                dp[i] += dp[i - j - 1]
    return dp[n - 1]

def is_valid(sequence):
    if False:
        for i in range(10):
            print('nop')
    ch = sequence[0]
    for c in sequence:
        if c != ch:
            return False
    if sequence == '0':
        return True
    if (ch >= '2' and ch <= '6' or ch == '8') and len(sequence) < 4:
        return True
    if ch == '7' or ch == '9':
        return True
    return False
print(num_smses('222'))
print(num_smses('2202222'))
print(num_smses('2222222222'))