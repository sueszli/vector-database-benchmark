"""
Create Palindrome (Minimum Insertions to Form a Palindrome)

Given a string, find the palindrome that can be made by inserting the fewest number of characters as possible anywhere in the word.
If there is more than one palindrome of minimum length that can be made, return the lexicographically earliest one (the first one alphabetically).

Input: 'race'
Output: 'ecarace'
Output explanation: Since we can add three letters to it (which is the smallest amount to make a palindrome).
                    There are seven other palindromes that can be made from "race" by adding three letters, but "ecarace" comes first alphabetically.

Input: 'google'
Output: 'elgoogle'

Input: 'abcda'
Output: 'adcbcda'
Output explanation: Number of insertions required is 2 - aDCbcda (between the first and second character).

Input: 'adefgfdcba'
Output: 'abcdefgfedcba'
Output explanation: Number of insertions required is 3 i.e. aBCdefgfEdcba.

=========================================
Recursive count how many insertions are needed, very slow and inefficient.
    Time Complexity:    O(2^N)
    Space Complexity:   O(N^2)  , for each function call a new string is created (and the recursion can have depth of max N calls)
Dynamic programming. Count intersections looking in 3 direction in the dp table (diagonally left-up or min(left, up)).
    Time Complexity:    O(N^2)
    Space Complexity:   O(N^2)
"""

def create_palindrome_1(word):
    if False:
        i = 10
        return i + 15
    n = len(word)
    if n == 1:
        return word
    if n == 2:
        if word[0] != word[1]:
            word += word[0]
        return word
    if word[0] == word[-1]:
        return word[0] + create_palindrome_1(word[1:-1]) + word[-1]
    first = create_palindrome_1(word[1:])
    first = word[0] + first + word[0]
    last = create_palindrome_1(word[:-1])
    last = word[-1] + last + word[-1]
    if len(first) < len(last):
        return first
    return last
import math

def create_palindrome_2(word):
    if False:
        return 10
    n = len(word)
    dp = [[0 for j in range(n)] for i in range(n)]
    for gap in range(1, n):
        left = 0
        for right in range(gap, n):
            if word[left] == word[right]:
                dp[left][right] = dp[left + 1][right - 1]
            else:
                dp[left][right] = min(dp[left][right - 1], dp[left + 1][right]) + 1
            left += 1
    return build_palindrome(word, dp, 0, n - 1)

def build_palindrome(word, dp, left, right):
    if False:
        print('Hello World!')
    if left > right:
        return ''
    if left == right:
        return word[left]
    if word[left] == word[right]:
        return word[left] + build_palindrome(word, dp, left + 1, right - 1) + word[left]
    if dp[left + 1][right] < dp[left][right - 1]:
        return word[left] + build_palindrome(word, dp, left + 1, right) + word[left]
    return word[right] + build_palindrome(word, dp, left, right - 1) + word[right]
word = 'race'
print(create_palindrome_1(word))
print(create_palindrome_2(word))
word = 'google'
print(create_palindrome_1(word))
print(create_palindrome_2(word))
word = 'abcda'
print(create_palindrome_1(word))
print(create_palindrome_2(word))
word = 'adefgfdcba'
print(create_palindrome_1(word))
print(create_palindrome_2(word))