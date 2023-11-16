"""
Valid Parentheses

Given a string of round, curly, and square open and closing brackets, return whether the brackets are balanced (well-formed).
For example, given the string '([])[]({})', you should return true.
Given the string '([)]' or '((()', you should return false.

Input: '()[{([]{})}]'
Output: True

=========================================
Use stack. Add open brackets in the stack, remove the last bracket from the stack if there is a closing brackets.
    Time Complexity:    O(N)
    Space Complexity:   O(N)
"""
from collections import deque

def is_valid(string):
    if False:
        return 10
    closing = {'}': '{', ']': '[', ')': '('}
    stack = deque()
    for char in string:
        if char in closing:
            if len(stack) == 0:
                return False
            last = stack.pop()
            if last != closing[char]:
                return False
        else:
            stack.append(char)
    return True
print(is_valid('()[{([]{})}]'))
print(is_valid('()[{([]{]})}]'))
print(is_valid('(]]])'))