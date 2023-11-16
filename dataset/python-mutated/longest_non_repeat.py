"""
Given a string, find the length of the longest substring
without repeating characters.

Examples:
Given "abcabcbb", the answer is "abc", which the length is 3.
Given "bbbbb", the answer is "b", with the length of 1.
Given "pwwkew", the answer is "wke", with the length of 3.
Note that the answer must be a substring,
"pwke" is a subsequence and not a substring.
"""

def longest_non_repeat_v1(string):
    if False:
        return 10
    '\n    Find the length of the longest substring\n    without repeating characters.\n    '
    if string is None:
        return 0
    dict = {}
    max_length = 0
    j = 0
    for i in range(len(string)):
        if string[i] in dict:
            j = max(dict[string[i]], j)
        dict[string[i]] = i + 1
        max_length = max(max_length, i - j + 1)
    return max_length

def longest_non_repeat_v2(string):
    if False:
        while True:
            i = 10
    '\n    Find the length of the longest substring\n    without repeating characters.\n    Uses alternative algorithm.\n    '
    if string is None:
        return 0
    (start, max_len) = (0, 0)
    used_char = {}
    for (index, char) in enumerate(string):
        if char in used_char and start <= used_char[char]:
            start = used_char[char] + 1
        else:
            max_len = max(max_len, index - start + 1)
        used_char[char] = index
    return max_len

def get_longest_non_repeat_v1(string):
    if False:
        while True:
            i = 10
    '\n    Find the length of the longest substring\n    without repeating characters.\n    Return max_len and the substring as a tuple\n    '
    if string is None:
        return (0, '')
    sub_string = ''
    dict = {}
    max_length = 0
    j = 0
    for i in range(len(string)):
        if string[i] in dict:
            j = max(dict[string[i]], j)
        dict[string[i]] = i + 1
        if i - j + 1 > max_length:
            max_length = i - j + 1
            sub_string = string[j:i + 1]
    return (max_length, sub_string)

def get_longest_non_repeat_v2(string):
    if False:
        while True:
            i = 10
    '\n    Find the length of the longest substring\n    without repeating characters.\n    Uses alternative algorithm.\n    Return max_len and the substring as a tuple\n    '
    if string is None:
        return (0, '')
    sub_string = ''
    (start, max_len) = (0, 0)
    used_char = {}
    for (index, char) in enumerate(string):
        if char in used_char and start <= used_char[char]:
            start = used_char[char] + 1
        elif index - start + 1 > max_len:
            max_len = index - start + 1
            sub_string = string[start:index + 1]
        used_char[char] = index
    return (max_len, sub_string)