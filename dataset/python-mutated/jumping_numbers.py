"""
Jumping numbers

A number is called as a Jumping Number if all adjacent digits in it differ by 1.
The difference between ‘9’ and ‘0’ is not considered as 1.
All single digit numbers are considered as Jumping Numbers.
For example 7, 8987 and 4343456 are Jumping numbers but 796 and 89098 are not.

Input: 20
Output: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 12]

=========================================
Make a tree (DFS way/backtracking), for each next digit take the last digit, go up and down
(example: 123, last digit is 3, so next digit should be 2 or 4).
    Time Complexity:    O(9 * 2^(NumOfDigits(N) - 1))
    Space Complexity:   O(1)        , recursion stack will have depth 9 (but this can be considered as constant)
"""

def jumping_numbers(x):
    if False:
        return 10
    result = []
    for i in range(1, 10):
        jumping_num(i, x, result)
    return result

def jumping_num(num, x, result):
    if False:
        return 10
    if num > x:
        return
    result.append(num)
    last_digit = num % 10
    next_num = num * 10
    if last_digit != 0:
        jumping_num(next_num + last_digit - 1, x, result)
    if last_digit != 9:
        jumping_num(next_num + last_digit + 1, x, result)
print(jumping_numbers(20))
print(jumping_numbers(100))