"""
Longest Common Substring

Given two strings X and Y, find the of longest common substring.

Input: 'GeeksforGeeks', 'GeeksQuiz'
Output: 'Geeks'

=========================================
Dynamic Programming Solution.
    Time Complexity:    O(N * M)
    Space Complexity:   O(M)
* For this problem exists a faster solution, using Suffix tree, Time Complexity O(N + M).
"""

def longest_common_substring(str1, str2):
    if False:
        while True:
            i = 10
    (n, m) = (len(str1), len(str2))
    curr = [0 for j in range(m + 1)]
    prev = []
    max_length = 0
    max_idx = 0
    for i in range(1, n + 1):
        prev = curr
        curr = [0 for j in range(m + 1)]
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                curr[j] = prev[j - 1] + 1
                if curr[j] > max_length:
                    max_length = curr[j]
                    max_idx = i
    return str1[max_idx - max_length:max_idx]
print(longest_common_substring('ABABC', 'BABCA'))
print(longest_common_substring('GeeksforGeeks', 'GeeksQuiz'))
print(longest_common_substring('abcdxyz', 'xyzabcd'))
print(longest_common_substring('zxabcdezy', 'yzabcdezx'))