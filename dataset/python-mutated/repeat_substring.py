"""
Given a non-empty string check if it can be constructed by taking
a substring of it and appending multiple copies of the substring together.

For example:
Input: "abab"
Output: True
Explanation: It's the substring "ab" twice.

Input: "aba"
Output: False

Input: "abcabcabcabc"
Output: True
Explanation: It's the substring "abc" four times.

Reference: https://leetcode.com/problems/repeated-substring-pattern/description/
"""

def repeat_substring(s):
    if False:
        while True:
            i = 10
    '\n    :type s: str\n    :rtype: bool\n    '
    str = (s + s)[1:-1]
    return s in str