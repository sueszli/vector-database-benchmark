"""
Longest Common Subsequence

Given 2 strings, find the longest common subseqence - https://en.wikipedia.org/wiki/Longest_common_subsequence_problem
NOT Longest Common Substring, this is a different problem.
Substring is a string composed ONLY of neighboring chars, subsequence could contain non-neighboring chars.

Input: 'ABAZDC', 'BACBAD'
Output: 'ABAD'

Input: 'I'm meto', 'I am Meto'
Output: 'Im eto'

=========================================
Dynamic programming solution.
Find more details here: https://www.geeksforgeeks.org/printing-longest-common-subsequence/
    Time Complexity:    O(N * M)
    Space Complexity:   O(N * M)    , can be O(M) see longest_common_substring.py solution (but you'll need to save subsequences)
"""

def longest_common_subsequence(str1, str2):
    if False:
        i = 10
        return i + 15
    (n, m) = (len(str1), len(str2))
    dp = [[0 for j in range(m + 1)] for i in range(n + 1)]
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if str1[i - 1] == str2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    letters = dp[n][m]
    result = ['' for i in range(letters)]
    i = n
    j = m
    while i != 0 and j != 0:
        if str1[i - 1] == str2[j - 1]:
            letters -= 1
            result[letters] = str1[i - 1]
            j -= 1
            i -= 1
        elif dp[i - 1][j] < dp[i][j - 1]:
            j -= 1
        else:
            i -= 1
    return ''.join(result)
print(longest_common_subsequence('ABAZDC', 'BACBAD'))
print(longest_common_subsequence("I'm meto", 'I am Meto'))