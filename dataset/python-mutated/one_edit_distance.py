"""
Given two strings S and T, determine if they are both one edit distance apart.
"""

def is_one_edit(s, t):
    if False:
        for i in range(10):
            print('nop')
    '\n    :type s: str\n    :type t: str\n    :rtype: bool\n    '
    if len(s) > len(t):
        return is_one_edit(t, s)
    if len(t) - len(s) > 1 or t == s:
        return False
    for i in range(len(s)):
        if s[i] != t[i]:
            return s[i + 1:] == t[i + 1:] or s[i:] == t[i + 1:]
    return True

def is_one_edit2(s, t):
    if False:
        i = 10
        return i + 15
    (l1, l2) = (len(s), len(t))
    if l1 > l2:
        return is_one_edit2(t, s)
    if len(t) - len(s) > 1 or t == s:
        return False
    for i in range(len(s)):
        if s[i] != t[i]:
            if l1 == l2:
                s = s[:i] + t[i] + s[i + 1:]
            else:
                s = s[:i] + t[i] + s[i:]
            break
    return s == t or s == t[:-1]