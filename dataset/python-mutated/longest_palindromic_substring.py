"""
Longest Palindromic Substring

Find the length of the longest palindromic substring.

Input: 'google'
Output: 4

=========================================
Simple algorithm, for each position compare left and right side and count the length of matching.
    Time Complexity:    O(N^2)
    Space Complexity:   O(1)
* For this problem exists a faster algorithm, called Manchester's Algorithm. Time Complexity O(N) and Space Complexity O(N).
"""

def longest_palindromic_substring(s):
    if False:
        return 10
    n = len(s)
    longest = 1
    for i in range(n):
        count_odd = compare_both_sides(s, 1, i - 1, i + 1)
        count_even = compare_both_sides(s, 0, i - 1, i)
        longest = max(longest, count_odd, count_even)
    return longest

def compare_both_sides(s, count, left, right):
    if False:
        for i in range(10):
            print('nop')
    n = len(s)
    while left >= 0 and right < n and (s[left] == s[right]):
        count += 2
        left -= 1
        right += 1
    return count
print(longest_palindromic_substring('google'))
print(longest_palindromic_substring('sgoaberebaogle'))
print(longest_palindromic_substring('abcdeef'))
print(longest_palindromic_substring('racecar'))
print(longest_palindromic_substring('abbabbc'))
print(longest_palindromic_substring('forgeeksskeegfor'))