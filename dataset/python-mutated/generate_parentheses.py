"""
Generate Parentheses

Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses.

Input: 3
Output:
        [
            '((()))',
            '(()())',
            '(())()',
            '()(())',
            '()()()'
        ]

=========================================
This problem could be solved in several ways (using stack, queue, or just a simple list - see letter_combinations.py), all of them have the same complexity.
I'll solve it using simple recursive algorithm.
    Time Complexity:    O(4^N)      , O(2^(2*N)) = O(4^N)
    Space Complexity:   O(4^N)
"""

def generate_parentheses(n):
    if False:
        i = 10
        return i + 15
    result = []
    if n == 0:
        return result
    combinations(result, n, n, '')
    return result

def combinations(result, open_left, close_left, combination):
    if False:
        return 10
    if close_left == 0:
        result.append(combination)
    elif open_left == 0:
        result.append(combination + ')' * close_left)
    else:
        combinations(result, open_left - 1, close_left, combination + '(')
        if open_left < close_left:
            combinations(result, open_left, close_left - 1, combination + ')')
print(generate_parentheses(3))