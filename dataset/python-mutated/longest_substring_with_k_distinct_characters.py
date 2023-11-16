"""
Longest Substring With k Distinct Characters

Given an integer k and a string s, find the length of the longest substring that contains at most k distinct characters.

Input: s = 'abcba', k = 2
Output: 'bcb'

=========================================
Simple solution (like sliding window or queue, add to the end and remove from the front).
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""

def longest_substring_with_distinct_characters(s, k):
    if False:
        i = 10
        return i + 15
    letters = {}
    longest = 0
    length = 0
    for i in range(len(s)):
        if s[i] in letters:
            letters[s[i]] += 1
            length += 1
        else:
            while len(letters) == k:
                firstLetter = s[i - length]
                letters[firstLetter] -= 1
                if letters[firstLetter] == 0:
                    del letters[firstLetter]
                length -= 1
            letters[s[i]] = 1
            length += 1
        longest = max(longest, length)
    return longest
print(longest_substring_with_distinct_characters('abcba', 2))
print(longest_substring_with_distinct_characters('abcbcbcbba', 2))