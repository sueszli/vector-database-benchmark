"""
Given two words word1 and word2, find the minimum number of steps required to
make word1 and word2 the same, where in each step you can delete one character
in either string.

For example:
Input: "sea", "eat"
Output: 2
Explanation: You need one step to make "sea" to "ea" and another step to make "eat" to "ea".

Reference: https://leetcode.com/problems/delete-operation-for-two-strings/description/
"""

def min_distance(word1, word2):
    if False:
        i = 10
        return i + 15
    '\n    Finds minimum distance by getting longest common subsequence\n\n    :type word1: str\n    :type word2: str\n    :rtype: int\n    '
    return len(word1) + len(word2) - 2 * lcs(word1, word2, len(word1), len(word2))

def lcs(word1, word2, i, j):
    if False:
        print('Hello World!')
    '\n    The length of longest common subsequence among the two given strings word1 and word2\n    '
    if i == 0 or j == 0:
        return 0
    if word1[i - 1] == word2[j - 1]:
        return 1 + lcs(word1, word2, i - 1, j - 1)
    return max(lcs(word1, word2, i - 1, j), lcs(word1, word2, i, j - 1))

def min_distance_dp(word1, word2):
    if False:
        return 10
    '\n    Finds minimum distance in a dynamic programming manner\n    TC: O(length1*length2), SC: O(length1*length2)\n\n    :type word1: str\n    :type word2: str\n    :rtype: int\n    '
    (length1, length2) = (len(word1) + 1, len(word2) + 1)
    res = [[0 for _ in range(length2)] for _ in range(length1)]
    if length1 == length2:
        for i in range(1, length1):
            (res[i][0], res[0][i]) = (i, i)
    else:
        for i in range(length1):
            res[i][0] = i
        for i in range(length2):
            res[0][i] = i
    for i in range(1, length1):
        for j in range(1, length2):
            if word1[i - 1] == word2[j - 1]:
                res[i][j] = res[i - 1][j - 1]
            else:
                res[i][j] = min(res[i - 1][j], res[i][j - 1]) + 1
    return res[len(word1)][len(word2)]