"""
Basic Calculator

Implement a basic calculator to evaluate a simple expression string.
The expression string may contain open '(' and closing parentheses ')',
the plus '+' or minus sign '-', non-negative integers and empty spaces ' '.

Input: '(1+(4+5+2)-3)+(6+8)'
Output: 23

Input: ' 2-1 + 2 '
Output: 3

=========================================
Start from the first character and respect the math rules. When brackets come, go inside the brackets
and compute the inner result, after that continue with adding or subtracting.
    Time Complexity:    O(N)
    Space Complexity:   O(K)    , much less than N (the deepest level of brackets)
"""

def basic_calculator(s):
    if False:
        return 10
    return calculate(s, 0)[0]

def calculate(s, i):
    if False:
        return 10
    sign = 1
    res = 0
    num = 0
    while i < len(s) and s[i] != ')':
        if s[i] >= '0' and s[i] <= '9':
            num = num * 10 + int(s[i])
        elif s[i] == '(':
            brackets = calculate(s, i + 1)
            res += brackets[0] * sign
            i = brackets[1]
        elif s[i] != ' ':
            res += num * sign
            num = 0
            if s[i] == '-':
                sign = -1
            elif s[i] == '+':
                sign = 1
        i += 1
    res += num * sign
    return (res, i)
print(basic_calculator('(1+(4+5+2)-3)+(6+8)'))
print(basic_calculator(' 2-1 + 2 '))