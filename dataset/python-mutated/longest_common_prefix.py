"""
Longest Common Prefix

Write a function to find the longest common prefix string amongst an array of strings.
If there is no common prefix, return an empty string ''.

Input: ['flower', 'flow', 'flight']
Output: 'fl'

Input: ['dog', 'racecar', 'car']
Output: ''

Input: ['aa', 'a']
Output: 'a'

=========================================
Many solutions for this problem exist (Divide and Conquer, Trie, etc) but this is the simplest and the fastest one.
Use the first string as LCP and iterate the rest in each step compare it with another one.
    Time Complexity:    O(N*A)  , N = number of strings, A = average chars, or simplest notation O(S) = total number of chars
    Space Complexity:   O(1)
"""

def longest_common_prefix(strs):
    if False:
        for i in range(10):
            print('nop')
    n = len(strs)
    if n == 0:
        return ''
    lcp = strs[0]
    lcp_idx = len(lcp)
    for i in range(1, n):
        lcp_idx = min(lcp_idx, len(strs[i]))
        for j in range(lcp_idx):
            if lcp[j] != strs[i][j]:
                lcp_idx = j
                break
    return lcp[:lcp_idx]
    "\n    # if you like string manipulations, you can use this code\n    # i don't like string manipulations in Python because they're immutable\n    lcp = strs[0]\n    for i in range(1, n):\n        lcp = lcp[:len(strs[i])]\n        for j in range(len(lcp)):\n            if lcp[j] != strs[i][j]:\n                lcp = lcp[:j]\n                break\n    return lcp\n    "
print(longest_common_prefix(['flower', 'flow', 'flight']))
print(longest_common_prefix(['dog', 'racecar', 'car']))
print(longest_common_prefix(['aa', 'a']))